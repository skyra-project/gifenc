/**
 * LZWEncoder
 *
 * Authors
 * - Kevin Weiner (original Java version - kweiner@fmsware.com)
 * - Thibault Imbert (AS3 version - bytearray.org)
 * - Johan Nordberg (JS version - code@johan-nordberg.com)
 * - Antonio Román (TS version - kyradiscord@gmail.com)
 *
 * Acknowledgements
 * - GIFCOMPR.C - GIF Image compression routines
 * - Lempel-Ziv compression based on 'compress'. GIF modifications by
 * - David Rowley (mgardi@watdcsu.waterloo.edu)
 *   GIF Image compression - modified 'compress'
 *   Based on: compress.c - File compression ala IEEE Computer, June 1984.
 *   By Authors:
 *   - Spencer W. Thomas (decvax!harpo!utah-cs!utah-gr!thomas)
 *   - Jim McKie (decvax!mcvax!jim)
 *   - Steve Davies (decvax!vax135!petsd!peora!srd)
 *   - Ken Turkowski (decvax!decwrl!turtlevax!ken)
 *   - James A. Woods (decvax!ihnp4!ames!jaw)
 *   - Joe Orost (decvax!vax135!petsd!joe)
 */

import type { ByteBuffer } from './ByteBuffer';

const EOF = -1;
const BITS = 12;
const HASH_SIZE = 5003; // 80% occupancy
const masks = [
	0x0000, 0x0001, 0x0003, 0x0007, 0x000f, 0x001f, 0x003f, 0x007f, 0x00ff, 0x01ff, 0x03ff, 0x07ff, 0x0fff, 0x1fff, 0x3fff, 0x7fff, 0xffff
];

export class LZWEncoder {
	public readonly width: number;
	public readonly height: number;
	private pixels: Uint8Array;
	private readonly initCodeSize: number;
	private currentAccumulator = 0;
	private currentBits = 0;
	private currentPixel = 0;
	private accumulator = 0;
	private firstUnusedEntry = 0; // first unused entry
	private maximumCode = 0;
	private remaining = 0;
	private bitSize = 0;

	// block compression parameters -- after all codes are used up,
	// and compression rate changes, start over.
	private clearFlag = false;

	// Algorithm: use open addressing double hashing (no chaining) on the
	// prefix code / next character combination. We do a variant of Knuth's
	// algorithm D (vol. 3, sec. 6.4) along with G. Knott's relatively-prime
	// secondary probe. Here, the modular division first probe is gives way
	// to a faster exclusive-or manipulation. Also do block compression with
	// an adaptive reset, whereby the code table is cleared when the compression
	// ratio decreases, but after the table fills. The variable-length output
	// codes are re-sized at this point, and a special CLEAR code is generated
	// for the decompression. Late addition: construct the table according to
	// file size for noticeable speed improvement on small files. Please direct
	// questions about this implementation to ames!jaw.
	private globalInitialBits = 0;
	private clearCode = 0;
	private endOfFrameCode = 0;

	private readonly accumulators = new Uint8Array(256);
	private readonly hashes = new Int32Array(HASH_SIZE);
	private readonly codes = new Int32Array(HASH_SIZE);

	public constructor(width: number, height: number, pixels: Uint8Array, colorDepth: number) {
		this.width = width;
		this.height = height;
		this.pixels = pixels;
		this.initCodeSize = Math.max(2, colorDepth);
	}

	public encode(outs: ByteBuffer) {
		outs.writeByte(this.initCodeSize); // write "initial code size" byte
		this.remaining = this.width * this.height; // reset navigation variables
		this.currentPixel = 0;
		this.compress(this.initCodeSize + 1, outs); // compress and write the pixel data
		outs.writeByte(0); // write block terminator
	}

	private compress(initialBits: number, output: ByteBuffer) {
		// Set up the globals: g_init_bits - initial number of bits
		this.globalInitialBits = initialBits;

		// Set up the necessary values
		this.clearFlag = false;
		this.bitSize = this.globalInitialBits;
		this.maximumCode = this.getMaximumCode(this.bitSize);

		this.clearCode = 1 << (initialBits - 1);
		this.endOfFrameCode = this.clearCode + 1;
		this.firstUnusedEntry = this.clearCode + 2;

		this.accumulator = 0; // clear packet

		let code = this.nextPixel();

		let hash = 80048;
		const hashShift = 4;
		const hashSizeRegion = HASH_SIZE;
		this.resetHashRange(hashSizeRegion); // clear hash table
		this.output(this.clearCode, output);

		let c: number;
		outerLoop: while ((c = this.nextPixel()) !== EOF) {
			hash = (c << BITS) + code;
			let i = (c << hashShift) ^ code; // xor hashing
			if (this.hashes[i] === hash) {
				code = this.codes[i];
				continue;
			}

			if (this.hashes[i] >= 0) {
				// non-empty slot
				let dispose = hashSizeRegion - i; // secondary hash (after G. Knott)
				if (i === 0) dispose = 1;
				do {
					if ((i -= dispose) < 0) i += hashSizeRegion;
					if (this.hashes[i] === hash) {
						code = this.codes[i];
						continue outerLoop;
					}
				} while (this.hashes[i] >= 0);
			}

			this.output(code, output);
			code = c;
			if (this.firstUnusedEntry < 1 << BITS) {
				this.codes[i] = this.firstUnusedEntry++; // code -> hashtable
				this.hashes[i] = hash;
			} else {
				this.clearCodeTable(output);
			}
		}

		// Put out the final code.
		this.output(code, output);
		this.output(this.endOfFrameCode, output);
	}

	// Add a character to the end of the current packet, and if it is 254
	// characters, flush the packet to disk.
	private addCharacter(c: number, outs: ByteBuffer) {
		this.accumulators[this.accumulator++] = c;
		if (this.accumulator >= 254) this.flushPacket(outs);
	}

	// Clear out the hash table
	// table clear for block compress
	private clearCodeTable(outs: ByteBuffer) {
		this.resetHashRange(HASH_SIZE);
		this.firstUnusedEntry = this.clearCode + 2;
		this.clearFlag = true;
		this.output(this.clearCode, outs);
	}

	// Reset code table
	private resetHashRange(hashSize: number) {
		this.hashes.fill(-1, 0, hashSize);
	}

	// Flush the packet to disk, and reset the accumulator
	private flushPacket(outs: ByteBuffer) {
		if (this.accumulator > 0) {
			outs.writeByte(this.accumulator);
			outs.fill(0, 0, this.accumulator);
			this.accumulator = 0;
		}
	}

	private getMaximumCode(size: number) {
		return (1 << size) - 1;
	}

	// Return the next pixel from the image
	private nextPixel() {
		if (this.remaining === 0) return EOF;

		--this.remaining;
		const pix = this.pixels[this.currentPixel++];
		return pix & 0xff;
	}

	private output(code: number, outs: ByteBuffer) {
		this.currentAccumulator &= masks[this.currentBits];

		if (this.currentBits > 0) this.currentAccumulator |= code << this.currentBits;
		else this.currentAccumulator = code;

		this.currentBits += this.bitSize;

		while (this.currentBits >= 8) {
			this.addCharacter(this.currentAccumulator & 0xff, outs);
			this.currentAccumulator >>= 8;
			this.currentBits -= 8;
		}

		// If the next entry is going to be too big for the code size,
		// then increase it, if possible.
		if (this.firstUnusedEntry > this.maximumCode || this.clearFlag) {
			if (this.clearFlag) {
				this.maximumCode = this.getMaximumCode((this.bitSize = this.globalInitialBits));
				this.clearFlag = false;
			} else {
				++this.bitSize;
				if (this.bitSize === BITS) this.maximumCode = 1 << BITS;
				else this.maximumCode = this.getMaximumCode(this.bitSize);
			}
		}

		if (code === this.endOfFrameCode) {
			// At EOF, write the rest of the buffer.
			while (this.currentBits > 0) {
				this.addCharacter(this.currentAccumulator & 0xff, outs);
				this.currentAccumulator >>= 8;
				this.currentBits -= 8;
			}
			this.flushPacket(outs);
		}
	}
}
