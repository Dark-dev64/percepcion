/* Prueba real de Chrome/AudioWorklet/HTTP con cámara y audio SIMULADOS.
 * Requiere scripts/serve_audio_smoke.py y Playwright en .test-runtime.
 * Por defecto simula YOLO en /api/frame. SMOKE_REAL_VIDEO=1 usa el motor
 * real del servidor indicado por SMOKE_BASE_URL; no valida precisión humana.
 */
const {chromium} = require('../.test-runtime/node_modules/playwright');
const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs');

async function waitUntil(check, timeout=30000) {
  const deadline=Date.now()+timeout;
  while (Date.now()<deadline) {
    if (await check()) return;
    await new Promise(resolve=>setTimeout(resolve,100));
  }
  throw new Error('No se cumplió la condición HTTP dentro del plazo');
}

(async () => {
  const base = process.env.SMOKE_BASE_URL || 'http://localhost:5057';
  const realVideo = process.env.SMOKE_REAL_VIDEO === '1';
  const screenshotPrefix = realVideo ? 'audio-real-video' : 'audio';
  const browser = await chromium.launch({
    executablePath:process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe',
    headless:true,
    args:['--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream','--autoplay-policy=no-user-gesture-required']
  });
  const page = await browser.newPage({viewport:{width:1400,height:1000}});
  const errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  page.on('console',message=>{ if (message.type()==='error') console.log('CONSOLE',message.text()); });
  let frames=0;
  let realPredictions=0;
  if (realVideo) {
    page.on('request',request=>{if(request.url().includes('/api/frame?')) frames++;});
    page.on('response',async response=>{
      if(response.url().includes('/api/frame?')) {
        const data=await response.json().catch(()=>({}));
        if(data.timing?.predict>0) realPredictions++;
      }
    });
  } else await page.route('**/api/frame?**', route => {
    frames++;
    return route.fulfill({json:{detections:[],summary:{level:'BAJO',n:0,message:'Señal de prueba; inferencia visual simulada'},riesgo:'BAJO',
      sistema:'OPERATIVO',timing:{predict:0,draw:0},pendiente:true,alertas:[],postura:[],obstaculos:[]}});
  });
  try {
    await page.goto(base);
    await page.locator('[data-p="cam"]').click();
    await page.locator('#bcam').click();
    await page.waitForFunction(()=>fallAudio.session !== null, null, {timeout:20000});
    const first=await page.evaluate(()=>fallAudio.session);
    await waitUntil(async()=>{
      const d=await (await page.request.get(`${base}/api/audio/sessions/${first}`)).json();
      return d.processed_chunks>=4;
    },120000);
    await page.waitForFunction(()=>/Analizando|silencio/.test(document.querySelector('#audio-status').innerText));
    await page.waitForFunction(()=>cam.ok);
    await page.waitForTimeout(500);
    console.log('BROWSER_STATE', await page.evaluate(()=>({camOn:cam.on,frames:cam.sent,busy:cam.busy,generation:camGeneration,
      error:cam.err,meta:document.querySelector('#cammeta').innerText,
      status:document.querySelector('#audio-status').innerText})), 'REQUESTS', frames, errors);
    assert.ok(frames>5,'video sigue enviando frames con audio activo');
    if(realVideo) assert.ok(realPredictions>0,'el servidor ejecuta inferencia visual real');
    const status=await page.locator('#audio-status').innerText();
    assert.match(status,/Analizando|silencio/);
    const onStart=frames, onClock=Date.now();
    await page.waitForTimeout(3000);
    const fpsAudioOn=(frames-onStart)*1000/(Date.now()-onClock);
    await page.screenshot({path:path.join(__dirname,`../.test-runtime/${screenshotPrefix}-smoke.png`),fullPage:true});
    await page.locator('#bstop').click();
    await waitUntil(async()=> (await page.request.get(`${base}/api/audio/sessions/${first}`)).status()===404);
    assert.equal(await page.evaluate(()=>fallAudio.resources),null);
    await page.locator('#bcam').click();
    await page.waitForFunction(old=>fallAudio.session && fallAudio.session!==old,first);
    const second=await page.evaluate(()=>fallAudio.session);
    await waitUntil(async()=>{
      const d=await (await page.request.get(`${base}/api/audio/sessions/${second}`)).json();
      return d.processed_chunks>=2;
    });
    await page.locator('#audio-detection').uncheck();
    assert.equal(await page.evaluate(()=>fallAudio.session),null);
    const before=frames;
    const offClock=Date.now();
    await page.waitForTimeout(3000);
    const fpsAudioOff=(frames-before)*1000/(Date.now()-offClock);
    assert.ok(frames>before,'video sigue al desactivar el detector');
    // Reemplazo de la pista simulada por una señal sintética local, dentro
    // del MISMO MediaStream de video; el detector y el backend son reales.
    const fixture = fs.readFileSync(path.join(__dirname,'../.test-runtime/fall-smoke.f32'));
    const pcm = Array.from(new Float32Array(fixture.buffer, fixture.byteOffset, fixture.length/4));
    await page.locator('#audio-detection').check();
    const alertFrameStart=frames, alertClock=Date.now();
    await page.evaluate(async samples => {
      fallAudio.stop();
      for (const track of cam.stream.getAudioTracks()) { cam.stream.removeTrack(track); track.stop(); }
      const context = new AudioContext();
      const source = context.createBufferSource();
      const buffer = context.createBuffer(1,samples.length,16000);
      buffer.copyToChannel(new Float32Array(samples),0);
      source.buffer = buffer;
      const destination = context.createMediaStreamDestination();
      source.connect(destination);
      cam.stream.addTrack(destination.stream.getAudioTracks()[0]);
      globalThis.smokeAudio = {context,source};
      await context.resume();
      await fallAudio.start(cam.stream);
      source.start();
    },pcm);
    const positive=await page.evaluate(()=>fallAudio.session);
    await waitUntil(async()=>{
      const data=await (await page.request.get(`${base}/api/alerts`)).json();
      return data.alertas.some(a=>a.stream_id===positive && a.type==='posible_caida');
    });
    await page.waitForFunction(id=>{
      const text=document.querySelector('#alertbox3').innerText;
      return text.includes(id) && text.includes('ALERTA: Se ha detectado un sonido compatible con una posible caída.');
    },positive,{timeout:10000});
    const fpsDuringAlert=(frames-alertFrameStart)*1000/(Date.now()-alertClock);
    const positiveState=await (await page.request.get(`${base}/api/audio/sessions/${positive}`)).json();
    await page.screenshot({path:path.join(__dirname,`../.test-runtime/${screenshotPrefix}-positive.png`),fullPage:true});
    await page.locator('#bstop').click();
    await page.evaluate(()=>smokeAudio.context.close());
    // Fuente sin audio: no se solicita micrófono alternativo.
    await page.evaluate(async()=>{
      const stream=new MediaStream();
      await fallAudio.start(stream);
    });
    assert.match(await page.locator('#audio-status').innerText(),/Sin pista/);
    assert.deepEqual(errors,[]);
    console.log(JSON.stringify({result:'PASS',frames,first,second,positive,status,pageErrors:errors,
      fpsAudioOn,fpsAudioOff,fpsDuringAlert,positiveState,
      videoInference:realVideo ? 'YOLO real en CPU' : 'simulada',realPredictions,
      media:'dispositivos simulados de Chrome'},null,2));
    if(realVideo) {
      const data=await (await page.request.get(`${base}/api/alerts`)).json();
      const ids=data.alertas.filter(a=>a.stream_id===positive).map(a=>a.alert_id);
      if(ids.length) await page.request.delete(`${base}/api/alertas`,{data:{ids}});
    }
  } finally {
    console.log('FINAL_UI',await page.evaluate(async()=>({errors:cam.err,alerts:document.querySelector('#alertbox3').innerText,
      audio:fallAudio.session ? await (await fetch(`/api/audio/sessions/${fallAudio.session}`)).json() : null})).catch(()=>null));
    await page.screenshot({path:path.join(__dirname,`../.test-runtime/${screenshotPrefix}-last.png`),fullPage:true}).catch(()=>{});
    await browser.close();
  }
})().catch(error=>{console.error(error);process.exitCode=1;});
