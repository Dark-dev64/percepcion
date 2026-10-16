/* Analiza exclusivamente las pistas del MediaStream que ya usa el video. */
class LiveFallAudio {
  constructor(onStatus) {
    this.onStatus = onStatus;
    this.generation = 0;
    this.session = null;
    this.resources = null;
    this.stream = null;
  }

  status(message) { this.onStatus(message); }

  closeSession(id) {
    if (id) fetch(`/api/audio/sessions/${id}/stop`, {method:'POST', keepalive:true}).catch(() => {});
  }

  stop(message = 'Análisis acústico detenido') {
    this.generation++;
    const old = this.resources;
    this.resources = null;
    this.closeSession(this.session);
    this.session = null;
    if (old) {
      clearInterval(old.timer);
      old.controller.abort();
      for (const [track, event, handler] of old.listeners) track.removeEventListener(event, handler);
      if (old.node) { old.node.port.onmessage = null; old.node.disconnect(); }
      if (old.source) old.source.disconnect();
      if (old.gain) old.gain.disconnect();
      old.context.close().catch(() => {});
    }
    this.status(message);
  }

  async start(stream) {
    this.stop();
    this.stream = stream;
    const tracks = stream.getAudioTracks().filter(t => t.readyState === 'live');
    if (!tracks.length) { this.status('Sin pista de audio; video disponible'); return; }
    const Context = globalThis.AudioContext || globalThis.webkitAudioContext;
    if (!Context || !globalThis.AudioWorkletNode) {
      this.status('El navegador no admite análisis acústico con AudioWorklet'); return;
    }
    const generation = this.generation;
    let context;
    try {
      context = new Context();
      const resources = {context, controller:new AbortController(), listeners:[], busy:false, timer:null};
      this.resources = resources;
      const current = () => generation === this.generation;
      this.status('Preparando detector acústico…');
      context.resume().catch(() => {});
      await context.audioWorklet.addModule('/static/audio-worklet.js');
      if (!current()) return;
      // No se pide otro micrófono: se usa la pista asociada al video.
      resources.source = context.createMediaStreamSource(new MediaStream(tracks));
      resources.node = new AudioWorkletNode(context, 'fall-audio');
      resources.gain = context.createGain();
      resources.gain.gain.value = 0;
      resources.source.connect(resources.node);
      resources.node.connect(resources.gain);
      resources.gain.connect(context.destination);
      const response = await fetch('/api/audio/sessions', {method:'POST', signal:resources.controller.signal});
      const info = await response.json();
      if (!current()) { this.closeSession(info.stream_id); return; }
      if (!response.ok) throw new Error(info.error || 'No se pudo iniciar el detector');
      this.session = info.stream_id;
      const id = this.session;
      let sequenceBase = null;
      const show = data => {
        if (!current()) return;
        if (data.state === 'error') throw new Error(data.error || 'Error del detector');
        const messages = {iniciando:'Cargando modelo acústico…', esperando_audio:'Esperando audio',
          analizando:'Analizando audio de esta transmisión', silencio:'Audio sin señal o en silencio; no permite asegurar cobertura'};
        this.status(context.state === 'suspended' ? 'Audio suspendido: pulsa «Reanudar audio»' :
          `${messages[data.state] || data.state} · ${data.dropped_chunks || 0} bloques descartados`);
      };
      const fail = error => {
        if (current() && error.name !== 'AbortError') this.stop(`Audio no disponible: ${error.message}. El video continúa.`);
      };
      resources.node.port.onmessage = async event => {
        const {buffer, sequence} = event.data;
        if (!current() || resources.busy || tracks.some(t => t.muted || !t.enabled)) return;
        if (sequenceBase === null) sequenceBase = sequence;
        const seq = sequence - sequenceBase;
        resources.busy = true;
        const timeout = setTimeout(() => resources.controller.abort(), 8000);
        try {
          const result = await fetch(`/api/audio/sessions/${id}/chunks`, {
            method:'POST', headers:{'Content-Type':'application/octet-stream',
              'X-Audio-Sequence':String(seq), 'X-Audio-Offset':String(seq * 8000)},
            body:buffer, signal:resources.controller.signal});
          const data = await result.json();
          if (!result.ok && result.status !== 429) throw new Error(data.error || 'Sesión de audio finalizada');
          show(data);
        } catch (error) {
          if (current() && error.name === 'AbortError') this.stop('El canal de audio no responde; vuelve a habilitar el análisis. El video continúa.');
          else fail(error);
        } finally {
          clearTimeout(timeout);
          resources.busy = false;
        }
      };
      for (const track of tracks) {
        const ended = () => this.stop('Pista de audio desconectada; video disponible');
        const muted = () => {
          resources.suspended = true;
          resources.node.port.onmessage = null;
          this.closeSession(this.session);
          this.session = null;
          this.status('Pista de audio interrumpida; esperando recuperación');
        };
        const unmuted = () => { if (current()) this.start(stream); };
        for (const [event, handler] of [['ended', ended], ['mute', muted], ['unmute', unmuted]]) {
          track.addEventListener(event, handler);
          resources.listeners.push([track, event, handler]);
        }
      }
      resources.timer = setInterval(async () => {
        if (!current()) return;
        if (tracks.some(t => t.muted || !t.enabled)) {
          resources.suspended = true;
          this.closeSession(this.session);
          this.session = null;
          this.status('Pista de audio silenciada o deshabilitada; sin cobertura acústica');
          return;
        }
        if (resources.suspended) { this.start(stream); return; }
        if (!this.session) return;
        try {
          const result = await fetch(`/api/audio/sessions/${id}`, {signal:resources.controller.signal});
          if (!result.ok) throw new Error('Sesión de audio finalizada');
          show(await result.json());
        } catch (error) { fail(error); }
      }, 3000);
      show(info);
    } catch (error) {
      if (generation === this.generation && error.name !== 'AbortError') {
        this.stop(`Audio no disponible: ${error.message}. El video continúa.`);
      }
    }
  }

  resume() {
    if (this.resources) this.resources.context.resume().catch(error => this.status(error.message));
  }
}
if (typeof module !== 'undefined') module.exports = {LiveFallAudio};
