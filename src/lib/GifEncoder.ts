import { Duplex, Readable } from 'stream';
import { types } from 'util';
import { ByteBuffer } from './ByteBuffer';
import { LZWEncoder } from './LZWEncoder';
import { NeuQuant } from './NeuQuant';

const NOP = () => {
	// no-op
};

const GIF_HEADER = new TextEncoder().encode('GIF89a');
const NETSCAPE_HEADER = new Uint8Array([0x4e, 0x45, 0x54, 0x53, 0x43, 0x41, 0x50, 0x45, 0x32, 0x2e, 0x30]); // NETSCAPE2.0

/**
 * The disposal method code.
 *
 * - `0`: No disposal specified. The decoder is not required to take any action.
 * - `1`: Do not dispose. The graphic is to be left in place.
 * - `2`: Restore to background color. The area used by the graphic must be restored to the background color.
 * - `3`: Restore to previous. The decoder is required to restore the area overwritten by the graphic with what was
 * there prior to rendering the graphic.
 * - `4` - `7`: To be defined.
 */
export type DisposalCode = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7;

export interface EncoderOptions {
	/**
	 * The frame delay in milliseconds.
	 * @default 0
	 */
	delay?: number;

	/**
	 * The frames per second, supersedes {@link EncoderOptions.delay} if set.
	 * @default 0
	 */
	framerate?: number;

	/**
	 * The GIF frame disposal code for the last added frame and any subsequent frames.
	 *
	 * Defaults to one of the following values:
	 * - `0` : If `transparent` is set
	 * - `2` : Otherwise
	 */
	dispose?: DisposalCode;

	/**
	 * The number of times to repeat the GIF, between `0` and `65536`, with two special cases:
	 * - `-1`: play once
	 * - `0`: repeat indefinitely
	 * @default -1
	 * @note When set to a value different to `-1`, the GIF will use the Netscape 2.0 extension.
	 */
	repeat?: number;

	/**
	 * The transparent color for the last added frame and any subsequent frames. Since all colors are subject to
	 * modification in the quantization process, the color in the final palette for each frame closest to the given
	 * color becomes the transparent color for that frame. May be set to null to indicate no transparent color.
	 */
	transparent?: number | null;

	/**
	 * The quality of color quantization (conversion of images to the maximum 256 colors allowed by the GIF
	 * specification) between `1` and `30`. Lower values (closer to 1) produce better colors but require significantly
	 * more resources and processing. `10` is the default value as it produces good color mapping at reasonable speeds.
	 *
	 * @note Values greater than 20 do not yield significant improvements in speed.
	 */
	quality?: number;
}

export class GifEncoder {
	/**
	 * The GIF image's width, between `1` and `65536`.
	 */
	public readonly width: number;

	/**
	 * The GIF image's height, between `1` and `65536`.
	 */
	public readonly height: number;

	/**
	 * The transparent color, `null` if no transparent color is given.
	 */
	private transparent: number | null = null;

	/**
	 * The transparent index in the color table.
	 */
	private transparentIndex = 0;

	/**
	 * Number between `-1` and `65536`, `-1` indicating no repeat (GIF89a specification), otherwise repeating `repeat`
	 * times with the exception of `0`, which repeats indefinitely.
	 */
	private repeat = -1;

	/**
	 * Frame delay in hundredths of a second (1 = 10ms).
	 */
	private delay = 0;

	private image: Uint8ClampedArray | null = null; // current frame
	private pixels: Uint8Array | null = null; // BGR byte array from frame
	private indexedPixels: Uint8Array | null = null; // converted frame indexed to palette
	private colorDepth: number | null = null; // number of bit planes
	private colorPalette: Float64Array | null = null; // RGB palette
	private usedEntry: boolean[] = []; // active palette entries
	private paletteSize = 7; // color table size (bits-1)
	private disposalMode: -1 | DisposalCode = -1; // disposal code (-1 = use default)
	private firstFrame = true;
	private sample = 10; // default sample interval for quantizer

	private started = false; // started encoding

	private readStreams: Readable[] = [];

	private out = new ByteBuffer();

	/**
	 * Constructs the GIF encoder.
	 * @param width An integer representing the GIF image's width, between `1` and `65536`.
	 * @param height An integer representing the GIF image's height, between `1` and `65536`.
	 */
	public constructor(width: number, height: number) {
		this.width = ~~width;
		this.height = ~~height;
	}

	/**
	 * Creates a readable stream and pushes it to the encoder's {@link GifEncoder.readStreams readable streams}.
	 * @returns The new readable stream.
	 * @example
	 * ```javascript
	 * const encoder = new GifEncoder(320, 240);
	 *
	 * // Stream the results as they are available into hello.gif
	 * encoder.createReadStream().pipe(fs.createWriteStream('hello.gif'));
	 * ```
	 */
	public createReadStream(): Readable;
	/**
	 * Uses an existing readable stream and pushes it to the encoder's {@link GifEncoder.readStreams readable streams}.
	 * @param readable The readable stream to use.
	 * @returns The given readable stream.
	 */
	public createReadStream<T extends Readable>(readable: T): T;
	public createReadStream(readable?: Readable) {
		if (!readable) {
			readable = new Readable();
			readable._read = NOP;
		}

		this.readStreams.push(readable);
		return readable;
	}

	public createWriteStream(options?: EncoderOptions): Duplex {
		if (options) {
			if (options.delay !== undefined) this.setDelay(options.delay);
			if (options.framerate !== undefined) this.setFramerate(options.framerate);
			if (options.dispose !== undefined) this.setDispose(options.dispose);
			if (options.repeat !== undefined) this.setRepeat(options.repeat);
			if (options.transparent !== undefined) this.setTransparent(options.transparent);
			if (options.quality !== undefined) this.setQuality(options.quality);
		}

		const duplex = new Duplex({ objectMode: true });
		duplex._read = NOP;
		this.createReadStream(duplex);

		duplex._write = (data, _enc, next) => {
			if (!this.started) this.start();
			this.addFrame(data);
			next();
		};

		const end = duplex.end.bind(duplex);
		duplex.end = (...args: readonly any[]) => {
			end(...args);
			this.finish();
		};

		return duplex;
	}

	public emit() {
		if (this.readStreams.length === 0) return;

		if (this.out.length) {
			const data = this.out.toArray();
			for (const stream of this.readStreams) {
				stream.push(data);
			}
		}
	}

	public end() {
		if (this.readStreams.length === 0) return;

		this.emit();
		for (const stream of this.readStreams) {
			stream.push(null);
		}

		this.readStreams = [];
	}

	/**
	 * Sets the delay time between each frame, or changes it for subsequent frames (applies to the next frame added)
	 */
	public setDelay(milliseconds: number) {
		this.delay = Math.round(milliseconds / 10);
	}

	/**
	 * Sets frame rate in frames per second.
	 */
	public setFramerate(fps: number) {
		this.delay = Math.round(100 / fps);
	}

	/**
	 * Sets the GIF frame disposal code for the last added frame and any subsequent frames.
	 *
	 * Defaults to one of the following values:
	 * - `0` : If `transparent` is set
	 * - `2` : Otherwise
	 *
	 * @see {@link DisposalCode}
	 */
	public setDispose(disposalCode: DisposalCode) {
		if (disposalCode >= 0) this.disposalMode = disposalCode;
	}

	/**
	 * Sets the number of times the set of GIF frames should be played.
	 * @param repeat The number of times between `-1` and `65536` to repeat the GIF, with two special cases:
	 * - `-1` (**default**): play once
	 * - `0`: repeat indefinitely
	 *
	 * @note This method has no effect after the first image was added.
	 */
	public setRepeat(repeat: number) {
		this.repeat = repeat;
	}

	/**
	 * Sets the transparent color for the last added frame and any subsequent frames. Since all colors are subject to
	 * modification in the quantization process, the color in the final palette for each frame closest to the given
	 * color becomes the transparent color for that frame. May be set to null to indicate no transparent color.
	 * @param color The color to be set in transparent pixels.
	 */
	public setTransparent(color: number | null) {
		this.transparent = color;
	}

	/**
	 * Sets the quality of color quantization (conversion of images to the maximum 256 colors allowed by the GIF
	 * specification). Lower values (`minimum` = 1) produce better colors, but slow processing significantly. `10` is
	 * the default, and produces good color mapping at reasonable speeds. Values greater than 20 do not yield
	 * significant improvements in speed.
	 * @param quality A number between `1` and `30`.
	 */
	public setQuality(quality: number) {
		if (quality < 1) quality = 1;
		this.sample = quality;
	}

	/**
	 * Adds the next GIF frame. The frame is not written immediately, but is actually deferred until the next frame is
	 * received so that timing data can be inserted. Calling {@link GifEncoder.finish} will flush all frames.
	 * @param imageData The image data to add into the next frame.
	 */
	public addFrame(imageData: CanvasRenderingContext2D | Uint8ClampedArray) {
		if (types.isUint8ClampedArray(imageData)) {
			this.image = imageData;
		} else {
			this.image = imageData.getImageData(0, 0, this.width, this.height).data;
		}

		this.getImagePixels(); // convert to correct format if necessary
		this.analyzePixels(); // build color table & map pixels

		if (this.firstFrame) {
			this.writeLogicalScreenDescriptor(); // logical screen descriptor
			this.writePalette(); // global color table
			if (this.repeat >= 0) {
				// use NS app extension to indicate reps
				this.writeNetscapeExtension();
			}
		}

		this.writeGraphicControlExtension(); // write graphic control extension
		this.writeImageDescriptor(); // image descriptor
		if (!this.firstFrame) this.writePalette(); // local color table
		this.writePixels(); // encode and write pixel data

		this.firstFrame = false;
		this.emit();
	}

	/**
	 * Adds final trailer to the GIF stream, if you don't call the finish method the GIF stream will not be valid.
	 */
	public finish() {
		this.out.writeByte(0x3b); // gif trailer
		this.end();
	}

	/**
	 * Writes the GIF file header
	 */
	public start() {
		this.out.writeBytes(GIF_HEADER);
		this.started = true;
		this.emit();
	}

	/**
	 * Analyzes current frame colors and creates a color map.
	 */
	private analyzePixels() {
		const pixels = this.pixels!;
		const len = pixels.length;
		const nPix = len / 3;

		this.indexedPixels = new Uint8Array(nPix);

		const quantifier = new NeuQuant(this.pixels!, this.sample);
		this.colorPalette = quantifier.getColorMap();

		// map image pixels to new palette
		let k = 0;
		for (let j = 0; j < nPix; j++) {
			const index = quantifier.lookupRGB(pixels[k++] & 0xff, pixels[k++] & 0xff, pixels[k++] & 0xff);
			this.usedEntry[index] = true;
			this.indexedPixels[j] = index;
		}

		this.pixels = null;
		this.colorDepth = 8;
		this.paletteSize = 7;

		// get closest match to transparent color if specified
		if (this.transparent !== null) {
			this.transparentIndex = this.findClosest(this.transparent);

			// ensure that pixels with full transparency in the RGBA image are using the selected transparent color index in the indexed image.
			for (let pixelIndex = 0; pixelIndex < nPix; pixelIndex++) {
				if (this.image![pixelIndex * 4 + 3] === 0) {
					this.indexedPixels[pixelIndex] = this.transparentIndex;
				}
			}
		}
	}

	/**
	 * Returns index of palette color closest to c.
	 * @param color The color to compare.
	 */
	private findClosest(color: number): number {
		if (this.colorPalette === null) return -1;

		const r = (color & 0xff0000) >> 16;
		const g = (color & 0x00ff00) >> 8;
		const b = color & 0x0000ff;
		let minimumIndex = 0;
		let distanceMinimum = 256 * 256 * 256;

		const len = this.colorPalette.length;
		for (let i = 0; i < len; ) {
			const index = i / 3;
			const dr = r - (this.colorPalette[i++] & 0xff);
			const dg = g - (this.colorPalette[i++] & 0xff);
			const db = b - (this.colorPalette[i++] & 0xff);
			const d = dr * dr + dg * dg + db * db;
			if (this.usedEntry[index] && d < distanceMinimum) {
				distanceMinimum = d;
				minimumIndex = index;
			}
		}

		return minimumIndex;
	}

	/**
	 * Updates {@link GifEncoder.pixels} by creating an RGB-formatted {@link Uint8Array} from the RGBA-formatted data.
	 */
	private getImagePixels(): void {
		const w = this.width;
		const h = this.height;
		this.pixels = new Uint8Array(w * h * 3);

		const data = this.image!;
		for (let i = 0, count = 0; i < h; i++) {
			for (let j = 0; j < w; j++) {
				const b = i * w * 4 + j * 4;
				this.pixels[count++] = data[b];
				this.pixels[count++] = data[b + 1];
				this.pixels[count++] = data[b + 2];
			}
		}
	}

	/**
	 * Writes the GCE (Graphic Control Extension).
	 */
	private writeGraphicControlExtension() {
		this.out.writeByte(0x21); // extension introducer
		this.out.writeByte(0xf9); // GCE label
		this.out.writeByte(4); // data block size

		let transparency: 0 | 1;
		let dispose: number;
		if (this.transparent === null) {
			transparency = 0;
			dispose = 0; // dispose = no action
		} else {
			transparency = 1;
			dispose = 2; // force clear if using transparent color
		}

		if (this.disposalMode >= 0) {
			dispose = this.disposalMode & 7; // user override
		}

		dispose <<= 2;

		// Write GCP's packed fields
		const fields =
			0b0000_0000 | // XXX0_0000 : Reserved
			dispose | //     000X_XX00 : Disposal Method
			0b0000_0000 | // 0000_00X0 : User Input Flag
			transparency; // 0000_000X : Transparent Color Flag

		this.out.writeByte(fields);

		this.writeShort(this.delay); // delay x 1/100 sec
		this.out.writeByte(this.transparentIndex); // transparent color index
		this.out.writeByte(0); // block terminator
	}

	/**
	 * Writes the ID (Image Descriptor).
	 */
	private writeImageDescriptor() {
		this.out.writeByte(0x2c); //     Image Descriptor block identifier
		this.writeShort(0); //           Image Left Position
		this.writeShort(0); //           Image Top Position
		this.writeShort(this.width); //  Image Width
		this.writeShort(this.height); // Image Height

		// Write the LCT (Local Color Table):
		const fields = this.firstFrame
			? 0b0000_0000 // The first frame uses the GCT (Global Color Table)
			: 0b1000_0000 | //     X000_0000 : Local Color Table Flag = 1
			  0b0000_0000 | //     0X00_0000 : Interlace Flag = 0
			  0b0000_0000 | //     00X0_0000 : Sort Flag = 0
			  0b0000_0000 | //     000X_X000 : Reserved
			  this.paletteSize; // 0000_0XXX : Size of Local Color Table

		this.out.writeByte(fields);
	}

	/**
	 * Writes the LSD (Logical Screen Descriptor)
	 */
	private writeLogicalScreenDescriptor() {
		// logical screen size
		this.writeShort(this.width);
		this.writeShort(this.height);

		// Write the GCT (Global Color Table):
		const fields =
			0b1000_0000 | // X000_0000     : GCT (Global Color Table) flag = 1
			0b0111_0000 | // 0XXX_0000     : Color Resolution = 7
			0b0000_0000 | // 0000_X000     : GCT sort flag = 0
			0b0000_0000 | // 0000_0X00     : Reserved
			this.paletteSize; // 0000_00XX : GCT (Global Color Table) size

		this.out.writeByte(fields);

		this.out.writeByte(0x000000); // background color index
		this.out.writeByte(0); // pixel aspect ratio - assume 1:1
	}

	/**
	 * Writes the Netscape application extension to define repeat count.
	 */
	private writeNetscapeExtension() {
		// Reference: http://www.vurdalakov.net/misc/gif/netscape-looping-application-extension

		this.out.writeByte(0x21); //             Extension
		this.out.writeByte(0xff); //             Application Extension
		this.out.writeByte(0x0b); //             Block Size
		this.out.writeBytes(NETSCAPE_HEADER); // Application Identifier + Application Authentication Code
		this.out.writeByte(0x03); //             Sub-block data size
		this.out.writeByte(0x01); //             Sub-block ID
		this.writeShort(this.repeat); //         Loop Count (up to 2 bytes, `0` = repeat forever)
		this.out.writeByte(0); //                Block Terminator
	}

	/**
	 * Writes the color table palette.
	 */
	private writePalette() {
		this.out.writeBytes(this.colorPalette!);
		this.out.writeTimes(0, 3 * 256 - this.colorPalette!.length);
	}

	private writeShort(pValue: number) {
		this.out.writeByte(pValue & 0xff);
		this.out.writeByte((pValue >> 8) & 0xff);
	}

	/**
	 * Encodes and writes pixel data into {@link GifEncoder.out}.
	 */
	private writePixels() {
		const enc = new LZWEncoder(this.width, this.height, this.indexedPixels!, this.colorDepth!);
		enc.encode(this.out);
	}
}
