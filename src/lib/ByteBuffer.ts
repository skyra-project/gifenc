export class ByteBuffer {
	private written = 0;
	private data: Uint8Array;

	/**
	 * Constructs the instance.
	 * @param size The amount of bytes to reserve, defaults to 8KB.
	 */
	public constructor(size = 8192) {
		this.data = new Uint8Array(size);
	}

	/**
	 * Gets the written data.
	 */
	public get length(): number {
		return this.written;
	}

	/**
	 * Resets the data.
	 * @note This does not de-allocate the data, instead, it sets the {@link ByteBuffer.written position} to zero.
	 */
	public reset(): void {
		this.written = 0;
	}

	/**
	 * Writes a single byte into the buffer.
	 * @param byte The byte to write, between `0x00` and `0xFF`.
	 */
	public writeByte(byte: number) {
		this.ensureByte();
		this.data[this.written++] = byte;
	}

	/**
	 * Writes the `byte` value `times` times.
	 * @param byte The byte to write `times` times.
	 * @param times The amount of times to write the `byte`.
	 */
	public writeTimes(byte: number, times: number): void {
		this.ensureBytes(times);

		for (let i = 0; i < times; i++) {
			this.data[this.written++] = byte;
		}
	}

	/**
	 * Writes `bytes` into the data.
	 * @param bytes The bytes to write.
	 */
	public writeBytes(bytes: ArrayLike<number>) {
		this.ensureBytes(bytes.length);

		// ArrayLike<T> does not define `Symbol.iterator`:
		// eslint-disable-next-line @typescript-eslint/prefer-for-of
		for (let i = 0; i < bytes.length; i++) {
			this.data[this.written++] = bytes[i];
		}
	}

	/**
	 * Gets a sub-array of what was written so far.
	 * @returns The written section of the data.
	 */
	public toArray(): Uint8Array {
		return this.data.subarray(0, this.written);
	}

	/**
	 * Fills the data with the `byte` value given a range.
	 * @param byte The value to write.
	 * @param start The start index, defaults to `0`.
	 * @param end The end index, defaults to {@link Uint8Array.length `this.data.length`}.
	 */
	public fill(byte: number, start?: number, end?: number) {
		this.data.fill(byte, start, end);
	}

	private ensureByte() {
		if (this.written + 1 >= this.data.length) {
			const size = this.data.length * 2;
			this.data = this.copyBytes(size);
		}
	}

	private ensureBytes(n: number) {
		if (this.written + n >= this.data.length) {
			const size = Math.pow(2, Math.ceil(Math.log(this.written + n) / Math.log(2)));
			this.data = this.copyBytes(size);
		}
	}

	private copyBytes(size: number) {
		const data = new Uint8Array(size);

		for (let i = 0; i < this.written; ++i) {
			data[i] = this.data[i];
		}

		return data;
	}
}
