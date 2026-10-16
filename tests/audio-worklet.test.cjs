const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

for (const rate of [16000, 44100, 48000, 96000]) {
  test(`worklet: mono, duración y secuencia a ${rate} Hz`, () => {
    let Processor;
    const sent = [];
    vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../app/static/audio-worklet.js'),'utf8'), {
      AudioWorkletProcessor:class { constructor() { this.port = {postMessage:data => sent.push(data)}; } },
      sampleRate:rate, Float32Array, Math,
      registerProcessor:(_, value) => { Processor = value; }
    });
    const processor = new Processor();
    for (let start=0; start<rate*2; start+=128) {
      const n = Math.min(128, rate*2-start);
      processor.process([[new Float32Array(n).fill(.2),new Float32Array(n).fill(.6)]], []);
    }
    assert.equal(sent.length, 4);
    for (let i=0;i<sent.length;i++) {
      assert.equal(sent[i].sequence,i);
      const pcm = new Float32Array(sent[i].buffer);
      assert.equal(pcm.length,8000);
      assert.ok(pcm.every(x => Math.abs(x-.4)<1e-5));
    }
  });
}

test('cliente sin pista de audio: no crea sesión ni pide micrófono', async () => {
  const {LiveFallAudio} = require('../app/static/audio-live.js');
  const states = [];
  const audio = new LiveFallAudio(s => states.push(s));
  await audio.start({getAudioTracks:()=>[]});
  assert.match(states.at(-1),/Sin pista de audio/);
  assert.equal(audio.resources,null);
  assert.equal(audio.session,null);
});
