/* Offline fake-DOM interaction checks, not browser or accessibility testing. */
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const assert = require('assert');
const root = path.resolve(__dirname, '..');
let downloads = [];
let blobs = [];
class Node {
  constructor(tag='div') { this.tag=tag; this.children=[]; this.attrs={}; this.events={}; this.dataset={}; this.value=''; this.textContent=''; this.hidden=false; this.className=''; this.classList={toggle:(name,on)=>{const set=new Set(this.className.split(' ').filter(Boolean));on?set.add(name):set.delete(name);this.className=[...set].join(' ');}}; }
  append(...nodes){this.children.push(...nodes);} replaceChildren(...nodes){this.children=[...nodes];} setAttribute(k,v){this.attrs[k]=v;} removeAttribute(k){delete this.attrs[k];}
  addEventListener(k,fn){this.events[k]=fn;} fire(k){assert(this.events[k],`${this.tag} missing ${k}`);return this.events[k]({target:this});}
  focus(){this.focused=true;} select(){this.selected=true;}
  click(){if(this.tag==='a') downloads.push({name:this.download,href:this.href});else this.fire('click');} remove(){}
}
const ids=['records-body','empty-state','record-detail','trace-subtitle','cases-list','case-detail','page-title','record-filter','reload','export-record','export-case','theme-marble','theme-basalt','copy-command','telegram-command','copy-status'];
const elements=Object.fromEntries(ids.map(id=>[id,new Node()]));
const panels=['records','synthetic','about'].map(name=>Object.assign(new Node('section'),{dataset:{panel:name}}));
const pages=['records','synthetic','about'].map(name=>Object.assign(new Node('button'),{dataset:{page:name}}));
const document={body:new Node('body'),documentElement:new Node('html'),getElementById:id=>{assert(elements[id],id);return elements[id];},createElement:tag=>new Node(tag),querySelectorAll:sel=>sel==='[data-panel]'?panels:sel==='[data-page]'?pages:[]};
let clipboardText=null; const navigator={clipboard:{writeText:async value=>{clipboardText=value;}}};elements['telegram-command'].value='/dpof status';
const ctx=vm.createContext({document,navigator,Blob:class{constructor(parts,opts){this.parts=parts;this.type=opts.type;blobs.push(this);}},URL:{createObjectURL:()=> 'blob:synthetic',revokeObjectURL:()=>{}},setTimeout:fn=>fn()});
for(const name of ['records_data.js','case_data.js','dashboard.js']){
 const code=fs.readFileSync(path.join(root,name),'utf8').replace(/^\uFEFF/,'');
 assert(!/\b(fetch|XMLHttpRequest|WebSocket|sendBeacon)\b/.test(code),name);
 vm.runInContext(code,ctx,{filename:name});
}
assert.equal(elements['records-body'].children.length,6);
assert(elements['record-detail'].children[2].children.some(node=>node.textContent==='Veto'));
elements['record-filter'].value='entry';elements['record-filter'].fire('change');assert.equal(elements['records-body'].children.length,2);
elements['record-filter'].value='review';elements['record-filter'].fire('change');assert.equal(elements['records-body'].children.length,4);assert(elements['record-detail'].children[2].children.some(node=>node.textContent==='Unknown'));
let row=elements['records-body'].children[1]; row.children[4].children[0].fire('click'); assert(elements['trace-subtitle'].textContent.includes('case A'));
elements['reload'].fire('click');assert.equal(elements['records-body'].children.length,6);assert.equal(elements.reload.textContent,'Snapshot reloaded');
pages[1].fire('click');assert.equal(panels[1].hidden,false);assert.equal(panels[0].hidden,true);assert.equal(pages[1].attrs['aria-current'],'page');
assert.equal(elements['cases-list'].children.length,4);
for(const [i,expected] of ['WOULD_DEFER','RECONFIRMED','INSUFFICIENT','NO_REENTRY_CONTEXT'].entries()){
 elements['cases-list'].children[i].fire('click');assert.equal(elements['case-detail'].children[0].textContent,expected);
}
elements['export-record'].fire('click');elements['export-case'].fire('click');assert.deepEqual(downloads.map(x=>x.name),['clio-recorded-event.json','clio-synthetic-case.json']);
const recordExport=JSON.parse(blobs[0].parts[0]), caseExport=JSON.parse(blobs[1].parts[0]);assert.equal(recordExport.label,'HISTORICAL_PAPER_NOT_LIVE');assert.equal(recordExport.record.action,'ENTER');assert.equal(caseExport.label,'SYNTHETIC_NOT_MARKET_DATA');assert.equal(caseExport.case.result.disposition,'NO_REENTRY_CONTEXT');
elements['theme-marble'].fire('click');assert.equal(document.documentElement.dataset.theme,'marble');assert.equal(elements['theme-marble'].attrs['aria-pressed'],'true');elements['theme-basalt'].fire('click');assert.equal(document.documentElement.dataset.theme,'basalt');
pages[2].fire('click');assert.equal(panels[2].hidden,false);assert.equal(panels[1].hidden,true);
(async()=>{await elements['copy-command'].fire('click');assert.equal(clipboardText,'/dpof status');assert(elements['copy-status'].textContent.includes('copied'));navigator.clipboard.writeText=async()=>{throw Error('denied');};await elements['copy-command'].fire('click');assert(elements['telegram-command'].selected);assert(elements['copy-status'].textContent.includes('manually'));console.log('PASS: 6 records; entry/review filters; selected trace; reload; 3 panes; 4 synthetic cases; 2 local exports; both themes; clipboard success/fallback; no network API in owned scripts');})().catch(e=>{console.error(e);process.exitCode=1;});
