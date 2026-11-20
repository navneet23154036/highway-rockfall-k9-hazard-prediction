'use strict';
const $=s=>document.querySelector(s),$$=s=>Array.from(document.querySelectorAll(s));
const form=$('#assessment-form');let schema=null,step=0,current=null,history=[];
try{history=JSON.parse(localStorage.getItem('rockfall_k9_history')||'[]');if(!Array.isArray(history))history=[];}catch{}
let endpoint=location.origin+'/predict';
const names={slope_height:'Slope height (ft)',slope_angle:'Slope angle (degrees)',slope_length:'Slope length (ft)',ditch_slope:'Ditch slope (degrees)',ditch_width:'Ditch width (ft)',ditch_depth:'Ditch depth (ft)',accidents:'Accidents',sight_dist:'Sight distance',speed:'Speed',adt:'Average daily traffic',avr:'AVR'};
const pretty=k=>schema?.item_names?.[k]||names[k]||(k.startsWith('sc')?'CDOT score '+k.slice(2):k.replaceAll('_',' '));
function field(key,score=false){const l=document.createElement('label');l.textContent=pretty(key);let el;if(score){el=document.createElement('select');const o=document.createElement('option');o.value='';o.textContent='Select score';el.append(o);for(const n of [0,3,9,27,81,243]){const opt=document.createElement('option');opt.value=n;opt.textContent='Score '+n;el.append(opt);}}else{el=document.createElement('input');el.type='number';el.min='0';el.step='any';el.placeholder='Enter value';if(['slope_angle','ditch_slope'].includes(key))el.max='90';}el.name=key;el.required=true;l.append(el);return l;}
function message(v){$('#form-message').textContent=v;$('#form-message').hidden=!v;}
function setStep(n){step=n;$$('.step').forEach((x,i)=>{x.classList.toggle('active',i===n);x.setAttribute('aria-selected',String(i===n));});$$('.step-panel').forEach((x,i)=>x.hidden=i!==n);$('#previous').hidden=n===0;$('#next').hidden=n===2;$('#run').hidden=n!==2;$('#next').textContent=n===0?'Continue to geometry':'Continue to ratings';}
function check(n){const bad=[...$(`[data-panel="${n}"]`).querySelectorAll('input,select')].find(x=>!x.checkValidity());if(bad){setStep(n);bad.reportValidity();return false;}return true;}
$$('.step').forEach((x,i)=>x.onclick=()=>setStep(i));$('#next').onclick=()=>{if(check(step))setStep(step+1);};$('#previous').onclick=()=>setStep(step-1);
function geology(){if(!schema)return;const rock=form.elements.rock_type.value,mode=rock.startsWith('Cryst')?'cr':rock.startsWith('Sed')?'sed':'blk';const target=$('#geology-fields');target.replaceChildren();for(const k of schema.geological_items[mode]||[])target.append(field(k,true));}
async function init(){try{const [h,r]=await Promise.all([fetch('/health'),fetch('/schema')]);if(!h.ok||!r.ok)throw Error('k=9 model not loaded. Export rockfall_v4_k9.joblib and restart Flask.');schema=await r.json();for(const key of ['rock_type','slope_type','region']){const select=form.elements[key];select.replaceChildren();const empty=document.createElement('option');empty.value='';empty.textContent='Select '+key.replaceAll('_',' ');select.append(empty);for(const v of schema.categories[key]){const opt=document.createElement('option');opt.value=v;opt.textContent=v;select.append(opt);}}const rat=form.elements.rater;rat.replaceChildren();for(const v of ['',...schema.raters]){const opt=document.createElement('option');opt.value=v;opt.textContent=v||'Unspecified';rat.append(opt);}for(const k of schema.geometry)$('#geometry-fields').append(field(k));for(const k of [...schema.environmental,...schema.derived,...schema.traffic])$('#rating-fields').append(field(k,true));const head=document.createElement('div');head.className='section-heading';head.innerHTML='<h3>Rock-specific geological scores</h3><span>9 selected items</span>';const grid=document.createElement('div');grid.className='field-grid';grid.id='geology-fields';$('#rating-fields').after(head,grid);form.elements.rock_type.addEventListener('change',geology);geology();renderInputGuide();message('k=9 model loaded. Enter real CDOT worksheet-compatible ratings.');}catch(e){message(e.message);$('#run').disabled=true;}}
// Demo presets populate actual form controls; classification always comes from /predict.
let presetBusy=false;
async function fillExample(kind){
 const status=(text)=>message(text);
 if(!schema){status('Model schema has not loaded. Check /health and /schema, then refresh.');return;}
 if(presetBusy)return;
 presetBusy=true;
 try{
  HTMLFormElement.prototype.reset.call(form);
  const assign=(name,value)=>{
   const el=form.elements.namedItem(name);
   if(!el)throw Error('Missing form parameter: '+name);
   el.value=String(value);
   if(el.value!==String(value))throw Error('Invalid option for '+name+': '+value);
   el.dispatchEvent(new Event('input',{bubbles:true}));
   return el;
  };
  const first=(key)=>{
   const values=schema.categories?.[key];
   if(!Array.isArray(values)||!values.length)throw Error('Missing '+key+' choices in model schema');
   return values[0];
  };
  assign('rock_type',first('rock_type'));
  assign('slope_type',first('slope_type'));
  assign('region',first('region'));
  geology();
  assign('assessment_name',kind==='low'?'DEMO - Lower-rating slope':'DEMO - Higher-rating slope');
  const geometry=kind==='low'
   ? {slope_height:35,slope_angle:30,slope_length:70,ditch_slope:10,ditch_width:20,ditch_depth:7}
   : {slope_height:150,slope_angle:75,slope_length:250,ditch_slope:55,ditch_width:4,ditch_depth:1};
  for(const [key,value] of Object.entries(geometry))assign(key,value);
  const rating=kind==='low'?3:243;
  for(const key of [...schema.environmental,...schema.derived,...schema.traffic])assign(key,rating);
  const geological=Array.from(document.querySelectorAll('#geology-fields select'));
  if(!geological.length)throw Error('Geological score fields did not load');
  for(const el of geological)assign(el.name,rating);
  // Show populated form first. The model, not the button name, decides the hazard class.
  setStep(2);
  current=null;$('#result-content').hidden=true;$('#empty-result').hidden=false;
  status((kind==='low'?'Low':'High')+'-rating demo populated: '+(Object.keys(geometry).length+schema.environmental.length+schema.derived.length+schema.traffic.length+geological.length)+' measurements/ratings. Running model...');
  await runPrediction(true);
 }catch(err){console.error('Preset error',err);status('Auto-fill error: '+err.message);}
 finally{presetBusy=false;}
}
const lowPresetButton=$('#sample-low');
const highPresetButton=$('#sample-high');
if(lowPresetButton)lowPresetButton.addEventListener('click',()=>fillExample('low'));
if(highPresetButton)highPresetButton.addEventListener('click',()=>fillExample('high'));
$('#reset').onclick=()=>{HTMLFormElement.prototype.reset.call(form);geology();current=null;$('#result-content').hidden=true;$('#empty-result').hidden=false;setStep(0);message('');};
form.addEventListener('submit',async e=>{e.preventDefault();await runPrediction(false);});
async function runPrediction(fromPreset=false){
 if(!schema){message('Model schema is still loading. Please try again.');return;}
 for(let n=0;n<3;n++)if(!check(n))return;
 const data=Object.fromEntries(new FormData(form));delete data.assessment_name;
 const button=$('#run');button.disabled=true;button.textContent='Assessing…';
 if(!fromPreset)message('');
 try{
  const res=await fetch(endpoint,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
  const obj=await res.json();if(!res.ok)throw Error(obj.error||`HTTP ${res.status}`);
  if(!['Low','Moderate','High'].includes(obj.hazard_class)||!Number.isFinite(obj.predicted_total_score))throw Error('Invalid k=9 API response');
  current={id:crypto.randomUUID(),time:new Date().toISOString(),name:form.elements.assessment_name.value||'Untitled slope',inputs:data,result:obj,mode:'v4 k=9 API'};
  renderResult();
  if(fromPreset)message('Demo inputs were filled. Actual model prediction: '+obj.hazard_class+' (score '+obj.predicted_total_score.toFixed(1)+'). This is illustrative, not a real site assessment.');
  $('#result-content').scrollIntoView({behavior:'smooth',block:'nearest'});
 }catch(e){message('Prediction could not be completed: '+e.message);}
 finally{button.disabled=false;button.textContent='Assess slope';}
}

function renderResult(){const r=current.result;$('#empty-result').hidden=true;$('#result-content').hidden=false;$('#result-tag').textContent='k=9 API result';$('#result-content').innerHTML=`<div class="result-hero"><div><small>PREDICTED HAZARD CLASS</small><strong class="hazard-${r.hazard_class.toLowerCase()}">${r.hazard_class}</strong></div><div><strong>${r.predicted_total_score.toFixed(1)}</strong><small>PREDICTED TOTAL SCORE</small></div></div><div class="bar-row"><div><span>Low cutoff</span><strong>${r.low_cutoff}</strong></div></div><div class="bar-row"><div><span>High cutoff</span><strong>${r.high_cutoff}</strong></div></div><div class="status-box ${r.hazard_class==='High'?'flagged':''}">${r.hazard_class==='High'?'Above the High-hazard threshold.':'Score-based hazard classification.'}</div><p class="result-source">v4 k=9 Gradient Boosting + Ridge ensemble. No calibrated k=9 class probability is available. This is not the probability of a future rockfall event.</p><details class="score-explanation"><summary>How is this hazard class calculated?</summary><div class="score-explanation-body"><p><strong>Step 1 — Predicted score:</strong> The v4 k=9 ensemble evaluates the submitted slope, environmental, traffic and geological ratings to estimate a total hazard score of <strong>${r.predicted_total_score.toFixed(1)}</strong>.</p><p><strong>Step 2 — Classification ranges:</strong></p><div class="score-range"><span>Low</span><b>Score ≤ ${r.low_cutoff}</b></div><div class="score-range"><span>Moderate</span><b>${r.low_cutoff} &lt; Score ≤ ${r.high_cutoff}</b></div><div class="score-range"><span>High</span><b>Score &gt; ${r.high_cutoff}</b></div><p><strong>Step 3 — Your result:</strong> ${r.hazard_class==='High'?`The score is ${(r.predicted_total_score-r.high_cutoff).toFixed(1)} points above the High cutoff (${r.high_cutoff}), so the classification is HIGH.`:r.hazard_class==='Low'?`The score is ${(r.low_cutoff-r.predicted_total_score).toFixed(1)} points below the Low cutoff (${r.low_cutoff}), so the classification is LOW.`:`The score is above ${r.low_cutoff} and at or below ${r.high_cutoff}, so the classification is MODERATE.`}</p><p class="score-caution">The score is not a percentage or calibrated rockfall probability. It is a model-based screening assessment, not a certified safety determination.</p></div></details><div class="result-actions"><button class="secondary" id="save-result">Save assessment</button><button class="secondary" id="download-result-pdf">Download PDF Report</button><button class="secondary" id="export-result">Export JSON</button></div>`;$('#save-result').onclick=save;$('#export-result').onclick=()=>download(current);$('#download-result-pdf').onclick=()=>downloadPDF([current]);}
async function downloadPDF(items){
 if(!items.length){alert('Save at least one assessment first.');return;}
 try{const response=await fetch('/report/pdf',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({assessments:items})});
 if(!response.ok){let msg='PDF report generation failed';try{msg=(await response.json()).error||msg;}catch{}throw Error(msg);}
 const blob=await response.blob(),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='rockfall-hazard-report-'+new Date().toISOString().slice(0,10)+'.pdf';document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),10000);
 }catch(err){alert('Unable to download PDF: '+err.message);}}
function save(){if(!current||history.some(x=>x.id===current.id))return;history.unshift(current);history=history.slice(0,100);try{localStorage.setItem('rockfall_k9_history',JSON.stringify(history));$('#save-result').textContent='Saved';$('#save-result').disabled=true;renderHistory();}catch{message('Storage unavailable. Export JSON instead.');}}
function download(v){const blob=new Blob([JSON.stringify(v,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='rockfall-k9-'+v.id.slice(0,8)+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function renderHistory(){$('#history-content').innerHTML=history.length?history.map((x,i)=>`<div class="history-record"><div><h3>${esc(x.name)}</h3><p>${esc(x.inputs.rock_type)} · ${esc(x.inputs.slope_type)} · Region ${esc(x.inputs.region)}</p><small>${new Date(x.time).toLocaleString()}</small></div><div class="record-right"><strong class="hazard-${x.result.hazard_class.toLowerCase()}">${x.result.hazard_class}</strong><button class="text-button history-pdf" data-index="${i}">Download PDF</button><button class="text-button history-report" data-index="${i}">Download HTML</button><button class="text-button history-export" data-index="${i}">Export JSON</button></div></div>`).join(''):'<div class="history-empty"><h2>Your assessments will appear here.</h2><p>Run and save a k=9 assessment.</p></div>';$$('.history-pdf').forEach(x=>x.onclick=()=>downloadPDF([history[Number(x.dataset.index)]]));$$('.history-export').forEach(x=>x.onclick=()=>download(history[Number(x.dataset.index)]));$$('.history-report').forEach(x=>x.onclick=()=>window.RockfallReports.download([history[Number(x.dataset.index)]],'single')); $('#clear-history').disabled=!history.length;}
$('#clear-history').onclick=()=>{if(confirm('Clear saved k=9 history?')){history=[];localStorage.removeItem('rockfall_k9_history');renderHistory();}};
const pages={assessment:'Slope assessment',insights:'Model insights',history:'Assessment history',project:'About the project',inputs:'About inputs'};function route(){const key=location.hash.slice(1),page=pages[key]?key:'assessment';$$('.page').forEach(x=>x.hidden=x.id!==page);$$('[data-page]').forEach(x=>{x.classList.toggle('active',x.dataset.page===page);if(x.dataset.page===page)x.setAttribute('aria-current','page');else x.removeAttribute('aria-current');});$('#breadcrumb').textContent=pages[page];document.title='Highway Rockfall Hazard Classification · '+pages[page];if(page==='history')renderHistory();window.scrollTo(0,0);}window.addEventListener('hashchange',route);route();renderHistory();

// Informational guide: labels come from the live model schema, while explanatory
// text below is plain-language context, not a substitute for CDOT score tables.
const guideDescriptions={
sc28:'Rockfall Frequency: how often rockfalls have been observed or documented at the site. Use historical inspection records.',
sc32:'Annual Precipitation: assessment of the site’s annual rainfall and snowfall exposure. Use official climate records.',
sc33:'Freeze / Thaw: exposure to freezing and thawing cycles that may affect cracks in rock.',
sc34:'Seepage / Water: observed water seepage or wet conditions on the slope face.',
sc35:'Slope Aspect: the compass direction the slope faces. Verify using a compass or aspect measurement.',
sc27:'Slope Height Score: a rating assigned to the measured slope height according to worksheet criteria.',
sc29:'Slope Angle Score: a rating assigned to measured slope steepness according to worksheet criteria.',
sc31:'Ditch Catchment: assessment of how effectively the roadside ditch catches falling rock debris.',
sc51:'Sight Distance: assessment of the visibility distance available to road users approaching a hazard.',
sc52:'Accident History: assessment of documented traffic accidents associated with the site. Use verified records.',
sc53:'Average Vehicle Risk: worksheet-based assessment of traffic exposure and vehicle risk.'
};
function guideCard(name,description){const el=document.createElement('article');el.className='card guide-card';const h=document.createElement('h3');h.textContent=name;const p=document.createElement('p');p.textContent=description;el.append(h,p);return el;}
function renderInputGuide(){if(!schema)return;const standard=$('#guide-standard'),geo=$('#guide-geology');if(!standard||!geo)return;standard.replaceChildren();for(const key of [...schema.environmental,...schema.derived,...schema.traffic]){standard.append(guideCard(pretty(key),guideDescriptions[key]||'A CDOT worksheet assessment parameter. Verify its precise definition and scoring ranges using the original scoring criteria.'));}geo.replaceChildren();for(const [mode,title] of [['cr','Crystalline'],['sed','Sedimentary'],['blk','Block in matrix']]){const group=document.createElement('article');group.className='card guide-card';const h=document.createElement('h3');h.textContent=title;const p=document.createElement('p');p.textContent='Applicable nine geological items (original worksheet labels):';const ul=document.createElement('ul');for(const key of schema.geological_items[mode]||[]){const li=document.createElement('li');li.textContent=pretty(key);ul.append(li);}const note=document.createElement('p');note.className='guide-muted';note.textContent='Assess rock structure, material and field conditions using the official CDOT scoring definitions. Do not infer a score from the label alone.';group.append(h,p,ul,note);geo.append(group);}}

init();

$('#download-history-report').onclick=()=>window.RockfallReports.download(history,'history');
$('#print-history-report').onclick=()=>window.RockfallReports.print(history);

const historyPDFButton=document.querySelector('#download-history-pdf');if(historyPDFButton)historyPDFButton.onclick=()=>downloadPDF(history);
