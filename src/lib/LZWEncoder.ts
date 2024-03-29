/**
 * LZWEncoder
 *
 * Authors
 * - Kevin Weiner (original Java version - kweiner@fmsware.com)
 * - Thibault Imbert (AS3 version - bytearray.org)
 * - Johan Nordberg (JS version - code@johan-nordberg.com)
 * - Aura Román (TS version - kyradiscord@gmail.com)
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
const masks = new Uint16Array([
	0x0000, 0x0001, 0x0003, 0x0007, 0x000f, 0x001f, 0x003f, 0x007f, 0x00ff, 0x01ff, 0x03ff, 0x07ff, 0x0fff, 0x1fff, 0x3fff, 0x7fff, 0xffff
]);

/**
 * @summary
 * Algorithm: use open addressing double hashing (no chaining) on the prefix code / next character combination.
 *
 * We do a variant of Knuth's algorithm D (vol. 3, sec. 6.4) along with G. Knott's relatively-prime secondary probe.
 * Here, the modular division first probe is gives way to a faster exclusive-or manipulation. Also do block compression
 * with an adaptive reset, whereby the code table is cleared when the compression ratio decreases, but after the table
 * fills. The variable-length output codes are re-sized at this point, and a special CLEAR code is generated for the
 * decompression.
 *
 * **Late addition**: construct the table according to file size for noticeable speed improvement on small files. Please
 * direct questions about this implementation to ames!jaw.
 */
export class LZWEncoder {
	/**
	 * The GIF image's width, between `1` and `65536`.
	 */
	public readonly width: number;

	/**
	 * The GIF image's height, between `1` and `65536`.
	 */
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

	private globalInitialBits = 0;
	private clearCode = 0;
	private endOfFrameCode = 0;

	private readonly accumulators = new Uint8Array(256);
	private readonly hashes = new Int32Array(HASH_SIZE);
	private readonly codes = new Int32Array(HASH_SIZE);

	/**
	 * Constructs a {@link LZWEncoder} instance.
	 * @param width The width of the image.
	 * @param height The height of the image.
	 * @param pixels The pixel data in RGB format.
	 * @param colorDepth The color depth.
	 */
	public constructor(width: number, height: number, pixels: Uint8Array, colorDepth: number) {
		this.width = width;
		this.height = height;
		this.pixels = pixels;
		this.initCodeSize = Math.max(2, colorDepth);
	}

	/**
	 * Encodes the image into the output.
	 * @param output The byte buffer to write to.
	 */
	public encode(output: ByteBuffer) {
		output.writeByte(this.initCodeSize); // write "initial code size" byte
		this.remaining = this.width * this.height; // reset navigation variables
		this.currentPixel = 0;
		this.compress(this.initCodeSize + 1, output); // compress and write the pixel data
		output.writeByte(0); // write block terminator
	}

	/**
	 * Compresses the GIF data.
	 * @param initialBits The initial bits for the compression.
	 * @param output The byte buffer to write to.
	 */
	private compress(initialBits: number, output: ByteBuffer) {
		// Set up the globals: globalInitialBits - initial number of bits
		this.globalInitialBits = initialBits;

		// Set up the necessary values
		this.clearFlag = false;
		this.bitSize = this.globalInitialBits;
		this.maximumCode = this.getMaximumCode(this.bitSize);

		this.clearCode = 1 << (initialBits - 1);
		this.endOfFrameCode = this.clearCode + 1;
		this.firstUnusedEntry = this.clearCode + 2;

		// Clear packet
		this.accumulator = 0;

		let code = this.nextPixel();

		let hash = 80048;
		const hashShift = 4;
		const hashSizeRegion = HASH_SIZE;
		this.resetHashRange(hashSizeRegion);
		this.processOutput(this.clearCode, output);

		let c: number;
		outerLoop: while ((c = this.nextPixel()) !== EOF) {
			hash = (c << BITS) + code;

			// XOR hashing:
			let i = (c << hashShift) ^ code;
			if (this.hashes[i] === hash) {
				code = this.codes[i];
				continue;
			}

			if (this.hashes[i] >= 0) {
				// Non-empty slot, perform secondary hash (after G. Knott):
				let dispose = hashSizeRegion - i;
				if (i === 0) dispose = 1;
				do {
					if ((i -= dispose) < 0) i += hashSizeRegion;
					if (this.hashes[i] === hash) {
						code = this.codes[i];
						continue outerLoop;
					}
				} while (this.hashes[i] >= 0);
			}

			this.processOutput(code, output);
			code = c;
			if (this.firstUnusedEntry < 1 << BITS) {
				// code -> hash-table
				this.codes[i] = this.firstUnusedEntry++;
				this.hashes[i] = hash;
			} else {
				this.clearCodeTable(output);
			}
		}

		// Put out the final code:
		this.processOutput(code, output);
		this.processOutput(this.endOfFrameCode, output);
	}

	/**
	 * Adds a character to the end of the current packet, and if it is at 254 characters, it flushes the packet to disk
	 * via {@link LZWEncoder.flushPacket}.
	 * @param c The character code to add.
	 * @param output The byte buffer to write to.
	 */
	private addCharacter(c: number, output: ByteBuffer) {
		this.accumulators[this.accumulator++] = c;
		if (this.accumulator >= 254) this.flushPacket(output);
	}

	/**
	 * Clears out the hash table for block compress.
	 * @param output The byte buffer to write to.
	 */
	private clearCodeTable(output: ByteBuffer) {
		this.resetHashRange(HASH_SIZE);
		this.firstUnusedEntry = this.clearCode + 2;
		this.clearFlag = true;
		this.processOutput(this.clearCode, output);
	}

	/**
	 * Resets the hash table given an amount of hashes.
	 * @param hashSize The amount of hashes to reset.
	 */
	private resetHashRange(hashSize: number) {
		this.hashes.fill(-1, 0, hashSize);
	}

	/**
	 * Flushes the packet to disk, and reset the accumulator.
	 * @param output The byte buffer to write to.
	 */
	private flushPacket(output: ByteBuffer) {
		if (this.accumulator > 0) {
			output.writeByte(this.accumulator);
			output.writeBytes(this.accumulators, 0, this.accumulator);
			this.accumulator = 0;
		}
	}

	/**
	 * Gets the maximum representable number for a given amount of bits.
	 * @param size The bit size to get the number from.
	 * @returns The maximum code given a number of bits.
	 * @example
	 * ```typescript
	 * getMaximumCode(6);
	 * // ➡ 0b0011_1111
	 * ```
	 */
	private getMaximumCode(size: number) {
		return (1 << size) - 1;
	}

	/**
	 * Gets the next pixel from the image.
	 * @returns The next pixel from the image.
	 */
	private nextPixel() {
		if (this.remaining === 0) return EOF;

		--this.remaining;
		const pixel = this.pixels[this.currentPixel++];
		return pixel & 0xff;
	}

	private processOutput(code: number, outs: ByteBuffer) {
		this.currentAccumulator &= masks[this.currentBits];
		this.currentAccumulator = this.currentBits > 0 ? (this.currentAccumulator |= code << this.currentBits) : code;

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
