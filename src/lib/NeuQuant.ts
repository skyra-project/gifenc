/**
 * NeuQuant Neural-Net Quantization Algorithm
 * ------------------------------------------
 *
 * Copyright (c) 1994 Anthony Dekker
 *
 * NEUQUANT Neural-Net quantization algorithm by Anthony Dekker, 1994.
 * See "Kohonen neural networks for optimal colour quantization"
 * in "Network: Computation in Neural Systems" Vol. 5 (1994) pp 351-367.
 * for a discussion of the algorithm.
 * See also  http://members.ozemail.com.au/~dekker/NEUQUANT.HTML
 *
 * Any party obtaining a copy of these files from the author, directly or
 * indirectly, is granted, free of charge, a full and unrestricted irrevocable,
 * world-wide, paid up, royalty-free, nonexclusive right and license to deal
 * in this software and documentation files (the "Software"), including without
 * limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons who receive
 * copies from any such party to do so, with the only requirement being
 * that this copyright notice remain intact.
 *
 * (JavaScript port 2012 by Johan Nordberg)
 * (TypeScript port 2021 by Antonio Román)
 */

/* eslint-disable prefer-destructuring, no-negated-condition */
const learningCycles = 100; // number of learning cycles
const maximumColorsSize = 256; // number of colors used
const maximumColorsPosition = maximumColorsSize - 1;

// defs for freq and bias
const networkBiasShift = 4; // bias for colour values
const integerBiasShift = 16; // bias for fractions
const integerBias = 1 << integerBiasShift;
const gammaShift = 10;
const betaShift = 10;
const beta = integerBias >> betaShift; /* beta = 1/1024 */
const betaGamma = integerBias << (gammaShift - betaShift);

// defs for decreasing radius factor
const maximumRadius = maximumColorsSize >> 3; // for 256 cols, radius starts
const initialRadiusBiasShift = 6; // at 32.0 biased by 6 bits
const initialRadiusBias = 1 << initialRadiusBiasShift;
const initialRadius = maximumRadius * initialRadiusBias; // and decreases by a
const initialRadiusDecrement = 30; // factor of 1/30 each cycle

// defs for decreasing alpha factor
const alphaBiasShift = 10; // alpha starts at 1.0
const initialAlpha = 1 << alphaBiasShift;

/* radbias and alpharadbias used for radpower calculation */
const radiusBiasShift = 8;
const radiusBias = 1 << radiusBiasShift;
const alphaRadiusBiasShift = alphaBiasShift + radiusBiasShift;
const alphaRadiusBias = 1 << alphaRadiusBiasShift;

// four primes near 500 - assume no image has a length so large that it is
// divisible by all four primes
const prime1 = 499;
const prime2 = 491;
const prime3 = 487;
const prime4 = 503;
const minimumPictureBytes = 3 * prime4;

export class NeuQuant {
	private pixels;
	private sampleFactorial;

	private networks!: Float64Array[]; // int[netsize][4]
	private networkIndex!: Int32Array; // for network lookup - really 256

	// bias and freq arrays for learning
	private biases!: Int32Array;
	private frequencies!: Int32Array;
	private radiusPower!: Int32Array;

	/**
	 * Creates the neural quantifier instance.
	 * @param pixels Array of pixels in RGB format, as such that it's decoded as `[r, g, b, r, g, b, r, g, b, ...]`.
	 * @param sampleFactorial Sampling factor from 1 to 30, where lower is better quality.
	 */
	public constructor(pixels: Uint8Array, sampleFactorial: number) {
		this.pixels = pixels;
		this.sampleFactorial = sampleFactorial;
		this.networks = [];
		this.networkIndex = new Int32Array(256);
		this.biases = new Int32Array(maximumColorsSize);
		this.frequencies = new Int32Array(maximumColorsSize);
		this.radiusPower = new Int32Array(maximumColorsSize >> 3);

		this.init();
		this.learn();
		this.unBiasNetwork();
		this.buildIndexes();
	}

	/*
    Method: getColormap
    builds colormap from the index
    returns array in the format:
    >
    > [r, g, b, r, g, b, r, g, b, ..]
    >
  */
	public getColorMap() {
		const map = new Float64Array(maximumColorsSize);
		const index = new Float64Array(maximumColorsSize);

		for (let i = 0; i < maximumColorsSize; i++) index[this.networks[i][3]] = i;

		let k = 0;
		for (let l = 0; l < maximumColorsSize; l++) {
			const network = this.networks[index[l]];
			map[k++] = network[0];
			map[k++] = network[1];
			map[k++] = network[2];
		}
		return map;
	}

	/**
	 * Searches for BGR values 0..255 and returns a color index
	 * @param b The blue color byte, between 0 and 255.
	 * @param g The green color byte, between 0 and 255.
	 * @param r The red color byte, between 0 and 255.
	 * @returns The best color index.
	 */
	public lookupRGB(b: number, g: number, r: number) {
		// Biggest possible distance is 256 * 3, so we will define the biggest as an out-of-bounds number.
		let bestDistance = 1000;
		let best = -1;

		const index = this.networkIndex[g];

		// Index on `g`
		for (let i = index; i < maximumColorsSize; ++i) {
			const network = this.networks[i];

			// Compare the distance of the green element, break if it's too big:
			let distance = network[1] - g;
			if (distance >= bestDistance) break;

			// If `distance` is negative, make it positive:
			if (distance < 0) distance = -distance;

			// Compare the distance with the blue element added, continue if it's too big:
			distance += Math.abs(network[0] - b);
			if (distance >= bestDistance) continue;

			// Compare the distance with the red element added, continue if it's too big:
			distance += Math.abs(network[2] - r);
			if (distance >= bestDistance) continue;

			bestDistance = distance;
			best = network[3];
		}

		// Start at networkIndex[g] and work outwards
		for (let j = index - 1; j >= 0; --j) {
			const network = this.networks[j];

			// Compare the distance of the green element, break if it's too big:
			let distance = g - network[1];
			if (distance >= bestDistance) break;

			// If `distance` is negative, make it positive:
			if (distance < 0) distance = -distance;

			// Compare the distance with the blue element added, continue if it's too big:
			distance += Math.abs(network[0] - b);
			if (distance >= bestDistance) continue;

			// Compare the distance with the red element added, continue if it's too big:
			distance += Math.abs(network[2] - r);
			if (distance >= bestDistance) continue;

			bestDistance = distance;
			best = network[3];
		}

		return best;
	}

	/**
	 * Initializes the state for the arrays.
	 */
	private init() {
		for (let i = 0; i < maximumColorsSize; i++) {
			const v = (i << (networkBiasShift + 8)) / maximumColorsSize;
			this.networks[i] = new Float64Array([v, v, v, 0]);
			this.frequencies[i] = integerBias / maximumColorsSize;
			this.biases[i] = 0;
		}
	}

	/**
	 * Un-biases network to give byte values 0..255 and record position i to prepare for sort.
	 */
	private unBiasNetwork() {
		for (let i = 0; i < maximumColorsSize; i++) {
			const network = this.networks[i];
			network[0] >>= networkBiasShift;
			network[1] >>= networkBiasShift;
			network[2] >>= networkBiasShift;
			network[3] = i; // record color number
		}
	}

	/**
	 * Moves neuron `i` towards biased (`B`, `G`, `R`) by factor `alpha`.
	 * @param alpha The factor at which the neuron `i` should move towards.
	 * @param i The neuron's index.
	 * @param b The blue color.
	 * @param g The green color.
	 * @param r The red color.
	 */
	private alterSingle(alpha: number, i: number, b: number, g: number, r: number) {
		const network = this.networks[i];
		network[0] -= (alpha * (network[0] - b)) / initialAlpha;
		network[1] -= (alpha * (network[1] - g)) / initialAlpha;
		network[2] -= (alpha * (network[2] - r)) / initialAlpha;
	}

	/**
	 * Moves neurons in a `radius` around index `i` towards biased (`B`, `G`, `R`) by factor
	 * {@link NeuQuant.radiusPower `radiusPower[m]`}.
	 * @param radius The radius around `i` to alter.
	 * @param i The neuron's index.
	 * @param b The blue color.
	 * @param g The green color.
	 * @param r The red color.
	 */
	private alterNeighbors(radius: number, i: number, b: number, g: number, r: number) {
		const lo = Math.abs(i - radius);
		const hi = Math.min(i + radius, maximumColorsSize);

		let j = i + 1;
		let k = i - 1;
		let m = 1;

		while (j < hi || k > lo) {
			const alpha = this.radiusPower[m++];

			if (j < hi) {
				const network = this.networks[j++];
				network[0] -= (alpha * (network[0] - b)) / alphaRadiusBias;
				network[1] -= (alpha * (network[1] - g)) / alphaRadiusBias;
				network[2] -= (alpha * (network[2] - r)) / alphaRadiusBias;
			}

			if (k > lo) {
				const network = this.networks[k--];
				network[0] -= (alpha * (network[0] - b)) / alphaRadiusBias;
				network[1] -= (alpha * (network[1] - g)) / alphaRadiusBias;
				network[2] -= (alpha * (network[2] - r)) / alphaRadiusBias;
			}
		}
	}

	/**
	 * Searches for biased BGR values.
	 *
	 * - Finds the closest neuron (minimum distance) and updates {@link NeuQuant.frequencies}.
	 * - Finds the best neuron (minimum distance-bias) and returns the position.
	 *
	 * For frequently chosen neurons, {@link NeuQuant.frequencies `frequencies[i]`} is high and
	 * {@link NeuQuant.biases `biases[i]`} is negative.
	 *
	 * The latter is determined by the multiplication of `gamma` with the subtraction of the inverse of
	 * {@link maximumColorsSize} with {@link NeuQuant.frequencies `frequencies[i]`}:
	 *
	 * ```typescript
	 * biases[i] = gamma * ((1 / maximumColorsSize) - frequencies[i])
	 * ```
	 * @param b The blue color.
	 * @param g The green color.
	 * @param r The red color.
	 * @returns The best bias position.
	 */
	private contest(b: number, g: number, r: number) {
		let bestDistance = ~(1 << 31);
		let bestBiasDistance = bestDistance;
		let bestPosition = -1;
		let bestBiasPosition = bestPosition;

		for (let i = 0; i < maximumColorsSize; i++) {
			const network = this.networks[i];

			const distance = Math.abs(network[0] - b) + Math.abs(network[1] - g) + Math.abs(network[2] - r);
			if (distance < bestDistance) {
				bestDistance = distance;
				bestPosition = i;
			}

			const biasDistance = distance - (this.biases[i] >> (integerBiasShift - networkBiasShift));
			if (biasDistance < bestBiasDistance) {
				bestBiasDistance = biasDistance;
				bestBiasPosition = i;
			}

			const betaFrequency = this.frequencies[i] >> betaShift;
			this.frequencies[i] -= betaFrequency;
			this.biases[i] += betaFrequency << gammaShift;
		}

		this.frequencies[bestPosition] += beta;
		this.biases[bestPosition] -= betaGamma;

		return bestBiasPosition;
	}

	/*
Private Method: inxbuild
sorts network and builds netindex[0..255]
*/
	private buildIndexes() {
		let j: number;
		let p: Float64Array;
		let q: Float64Array;
		let smallpos: number;
		let smallval: number;
		let previouscol = 0;
		let startpos = 0;
		for (let i = 0; i < maximumColorsSize; i++) {
			p = this.networks[i];
			smallpos = i;
			[, smallval] = p; // index on g
			// find smallest in i..netsize-1
			for (j = i + 1; j < maximumColorsSize; j++) {
				q = this.networks[j];
				if (q[1] < smallval) {
					// index on g
					smallpos = j;
					[, smallval] = q; // index on g
				}
			}
			q = this.networks[smallpos];
			// swap p (i) and q (smallpos) entries
			if (i !== smallpos) {
				[q[0], p[0]] = [p[0], q[0]];
				[q[1], p[1]] = [p[1], q[1]];
				[q[2], p[2]] = [p[2], q[2]];
				[q[3], p[3]] = [p[3], q[3]];
			}
			// smallval entry is now in position i

			if (smallval !== previouscol) {
				this.networkIndex[previouscol] = (startpos + i) >> 1;
				for (j = previouscol + 1; j < smallval; j++) this.networkIndex[j] = i;
				previouscol = smallval;
				startpos = i;
			}
		}
		this.networkIndex[previouscol] = (startpos + maximumColorsPosition) >> 1;
		for (j = previouscol + 1; j < 256; j++) this.networkIndex[j] = maximumColorsPosition; // really 256
	}

	/**
	 * Runs the main learning loop.
	 */
	private learn() {
		let i;

		const length = this.pixels.length;
		const alphaDecrement = 30 + (this.sampleFactorial - 1) / 3;
		const samplePixels = length / (3 * this.sampleFactorial);
		let delta = ~~(samplePixels / learningCycles);
		let alpha = initialAlpha;
		let radius = initialRadius;

		let rad = radius >> initialRadiusBiasShift;

		if (rad <= 1) rad = 0;
		for (i = 0; i < rad; i++) this.radiusPower[i] = alpha * (((rad * rad - i * i) * radiusBias) / (rad * rad));

		let step;
		if (length < minimumPictureBytes) {
			this.sampleFactorial = 1;
			step = 3;
		} else if (length % prime1 !== 0) {
			step = 3 * prime1;
		} else if (length % prime2 !== 0) {
			step = 3 * prime2;
		} else if (length % prime3 !== 0) {
			step = 3 * prime3;
		} else {
			step = 3 * prime4;
		}

		let pixelPosition = 0; // current pixel

		i = 0;
		while (i < samplePixels) {
			const b = (this.pixels[pixelPosition] & 0xff) << networkBiasShift;
			const g = (this.pixels[pixelPosition + 1] & 0xff) << networkBiasShift;
			const r = (this.pixels[pixelPosition + 2] & 0xff) << networkBiasShift;

			let j = this.contest(b, g, r);

			this.alterSingle(alpha, j, b, g, r);
			if (rad !== 0) this.alterNeighbors(rad, j, b, g, r); // alter neighbors

			pixelPosition += step;
			if (pixelPosition >= length) pixelPosition -= length;

			i++;

			if (delta === 0) delta = 1;
			if (i % delta === 0) {
				alpha -= alpha / alphaDecrement;
				radius -= radius / initialRadiusDecrement;
				rad = radius >> initialRadiusBiasShift;

				if (rad <= 1) rad = 0;
				for (j = 0; j < rad; j++) this.radiusPower[j] = alpha * (((rad * rad - j * j) * radiusBias) / (rad * rad));
			}
		}
	}
}
