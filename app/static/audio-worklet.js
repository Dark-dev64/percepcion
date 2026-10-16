/* PCM mono a 16 kHz con integración de área para 44.1/48/96 kHz.
 * Un bloque transferible de 500 ms; no trabajo de red en el hilo de audio.
 */
class FallAudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.remaining = this.ratio;
    this.sum = 0;
    this.block = new Float32Array(8000);
    this.position = 0;
    this.sequence = 0;
  }

  process(inputs, outputs) {
    const channels = inputs[0];
    for (const output of outputs) for (const channel of output) channel.fill(0);
    if (!channels || !channels.length) return true;
    for (let i = 0; i < channels[0].length; i++) {
      let mono = 0;
      for (const channel of channels) mono += channel[i] / channels.length;
      let available = 1;
      while (available > 1e-10) {
        const portion = Math.min(available, this.remaining);
        this.sum += mono * portion;
        this.remaining -= portion;
        available -= portion;
        if (this.remaining < 1e-10) {
          this.block[this.position++] = Math.max(-1, Math.min(1, this.sum / this.ratio));
          this.remaining = this.ratio;
          this.sum = 0;
          if (this.position === this.block.length) {
            const buffer = this.block.buffer;
            this.port.postMessage({buffer, sequence: this.sequence++}, [buffer]);
            this.block = new Float32Array(8000);
            this.position = 0;
          }
        }
      }
    }
    return true;
  }
}
registerProcessor('fall-audio', FallAudioProcessor);
