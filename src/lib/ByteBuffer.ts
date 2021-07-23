export class ByteBuffer {
	private written = 0;
	private data: Uint8Array;

	public constructor(size = 16) {
		this.data = new Uint8Array(size);
	}

	public get length(): number {
		return this.written;
	}

	public clear(): void {
		this.written = 0;
	}

	public writeByte(byte: number) {
		this.ensureByte();
		this.data[this.written++] = byte;
	}

	public writeTimes(byte: number, times: number): void {
		this.ensureBytes(times);

		for (let i = 0; i < times; i++) {
			this.data[this.written++] = byte;
		}
	}

	public writeBytes(bytes: ArrayLike<number>) {
		this.ensureBytes(bytes.length);

		// ArrayLike<T> does not define `Symbol.iterator`:
		// eslint-disable-next-line @typescript-eslint/prefer-for-of
		for (let i = 0; i < bytes.length; i++) {
			this.data[this.written++] = bytes[i];
		}
	}

	public toArray(): Uint8Array {
		return this.data.subarray(0, this.written);
	}

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
