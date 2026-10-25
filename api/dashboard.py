"""Single-page dashboard served at /app. Plain HTML + JS, no build step; it calls the API on the same host."""

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PlayerPulse</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#16181d;--muted:#5f6672;--line:#e3e6eb;--accent:#2f5bd3;
--low:#1f8a4c;--med:#b7791f;--high:#c53030;--bar:#dfe5f3}
@media (prefers-color-scheme:dark){:root{--bg:#111318;--card:#1a1d24;--ink:#e8eaee;--muted:#9aa1ad;--line:#2a2f39;
--accent:#7c9cff;--low:#4cc38a;--med:#e0a84a;--high:#f26b6b;--bar:#2a3550}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1080px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:26px;margin:0}
h2{font-size:16px;margin:0 0 12px}
.sub{color:var(--muted);margin:4px 0 20px}
a{color:var(--accent)}
.grid{display:grid;gap:16px;grid-template-columns:1fr}
@media(min-width:860px){.grid.two{grid-template-columns:1.1fr .9fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:16px}
.stat .v{font-size:22px;font-weight:650}
.stat .k{color:var(--muted);font-size:13px}
form{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
label{font-size:12px;color:var(--muted);display:flex;flex-direction:column;gap:3px}
input,select{font:inherit;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--ink)}
.row{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 12px}
button{font:inherit;padding:7px 12px;border-radius:6px;border:1px solid var(--line);background:var(--bg);color:var(--ink);cursor:pointer}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:600}
.result .p{font-size:40px;font-weight:700}
.badge{display:inline-block;padding:2px 10px;border-radius:999px;font-weight:600;font-size:13px;color:#fff}
.meter{height:10px;background:var(--bar);border-radius:5px;position:relative;margin:10px 0 4px}
.meter i{position:absolute;left:0;top:0;bottom:0;border-radius:5px}
.meter b{position:absolute;top:-4px;bottom:-4px;width:2px;background:var(--ink)}
.small{font-size:12px;color:var(--muted)}
ul.warn{padding-left:18px;margin:8px 0 0;color:var(--med)}
table{width:100%;border-collapse:collapse;font-size:14px}
th,td{text-align:left;padding:6px 4px;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-weight:500}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.hbar{height:8px;background:var(--accent);border-radius:4px}
.tabs{display:flex;gap:6px;margin-bottom:10px}
.tabs button.on{border-color:var(--accent);color:var(--accent);font-weight:600}
</style>
</head>
<body>
<main>
<h1>PlayerPulse</h1>
<p class="sub">Which engaged World of Warcraft players will not come back next month? XGBoost trained on 10.8M activity logs (Jan–Jul 2008), tested on Sep–Nov.
<a href="/docs">API docs</a> · <a href="https://github.com/SanjayNalluri/playerpulse-churn-prediction">GitHub</a></p>

<div class="stats" id="stats"></div>

<div class="grid two">
<section class="card">
<h2>Score a player</h2>
<div class="row">
<button type="button" data-preset="risky">Fading player</button>
<button type="button" data-preset="active">Active raider</button>
<button type="button" data-preset="dk">New Death Knight (lvl 72)</button>
</div>
<form id="f"></form>
<div class="row" style="margin-top:12px"><button class="primary" id="go" type="button">Predict churn</button></div>
</section>

<section class="card result" id="res">
<h2>Result</h2>
<p class="small">Pick a preset or edit the fields, then press Predict.</p>
</section>
</div>

<section class="card" style="margin-top:16px">
<h2>Subgroup performance audit (test months)</h2>
<p class="small" style="margin-top:-6px">Share of real churners the model catches, by in-game group. Level 71–80 and Death Knights arrived with the November expansion, after training, which is why they score lowest.</p>
<div class="tabs" id="tabs"></div>
<table id="fair"></table>
</section>
</main>

<script>
const FIELDS=[["charclass","Class","sel"],["level_end","Level at month end"],["levels_gained","Levels gained"],
["hours_played","Hours played"],["active_days","Active days"],["n_zones","Zones visited"],["n_guilds","Guilds"],
["share_city","Share in cities (0–1)"],["share_dungeon","Share in dungeons"],["share_battleground","Share in battlegrounds"],
["share_arena","Share in arenas"],["share_in_guild","Share in a guild"],["days_since_last_seen","Days since last seen"],
["tenure_days","Days since first seen"],["prev_hours","Hours last month"],["hours_change","Change in hours"]];
const CLASSES=["Death Knight","Druid","Hunter","Mage","Paladin","Priest","Rogue","Shaman","Warlock","Warrior"];
const PRESETS={
risky:{charclass:"Warrior",level_end:70,levels_gained:0,hours_played:6,active_days:2,n_zones:4,n_guilds:1,share_city:0.4,share_dungeon:0,share_battleground:0,share_arena:0,share_in_guild:0.5,days_since_last_seen:20,tenure_days:200,prev_hours:40,hours_change:-34},
active:{charclass:"Mage",level_end:70,levels_gained:0,hours_played:60,active_days:25,n_zones:30,n_guilds:1,share_city:0.3,share_dungeon:0.25,share_battleground:0.05,share_arena:0.05,share_in_guild:0.9,days_since_last_seen:1,tenure_days:200,prev_hours:50,hours_change:10},
dk:{charclass:"Death Knight",level_end:72,levels_gained:17,hours_played:30,active_days:10,n_zones:25,n_guilds:1,share_city:0.2,share_dungeon:0.05,share_battleground:0,share_arena:0,share_in_guild:0.6,days_since_last_seen:3,tenure_days:15,prev_hours:0,hours_change:30}};
const f=document.getElementById("f");
f.innerHTML=FIELDS.map(([k,l,t])=>t?`<label>${l}<select name="${k}">${CLASSES.map(c=>`<option>${c}</option>`).join("")}</select></label>`
:`<label>${l}<input name="${k}" type="number" step="any"></label>`).join("");
function fill(p){for(const k in p)f.elements[k].value=p[k]}
fill(PRESETS.risky);
document.querySelectorAll("[data-preset]").forEach(b=>b.onclick=()=>{fill(PRESETS[b.dataset.preset]);predict()});
let T=0.2;
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v);
let reqId=0;  // only the latest request may update the result (an older reply can arrive later)
async function predict(){
 const my=++reqId;
 const body={};FIELDS.forEach(([k,,t])=>body[k]=t?f.elements[k].value:Number(f.elements[k].value));
 const res=document.getElementById("res");
 const r=await fetch("/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
 const j=await r.json();
 if(my!==reqId)return;
 if(!r.ok){res.innerHTML=`<h2>Result</h2><p class="small">Input rejected: ${(j.detail||[]).map(d=>d.loc.slice(-1)[0]+" "+d.msg).join("; ")}</p>`;return}
 const col=css("--"+(j.risk==="medium"?"med":j.risk)),pc=(j.churn_probability*100).toFixed(1);
 res.innerHTML=`<h2>Result</h2><div class="p">${pc}%</div><div class="small">chance this player does not return next month</div>
 <div class="meter"><i style="width:${pc}%;background:${col}"></i><b style="left:${T*100}%"></b></div>
 <div class="small">Line = decision threshold (${T})</div>
 <p><span class="badge" style="background:${col}">${j.risk} risk</span> &nbsp;${j.will_churn?"Flag for a retention offer":"No action needed"}</p>
 ${j.warnings.length?`<div class="small">Warnings</div><ul class="warn">${j.warnings.map(w=>`<li>${w}</li>`).join("")}</ul>`:`<p class="small">No warnings: inputs look like the training data.</p>`}`;
}
document.getElementById("go").onclick=predict;
fetch("/model").then(r=>r.json()).then(m=>{T=m.threshold;const t=m.test_metrics;
 if(m.classes){const sel=f.elements.charclass,cur=sel.value,all=[...new Set([...m.classes,"Death Knight"])].sort();
  sel.innerHTML=all.map(c=>`<option>${c}</option>`).join("");sel.value=all.includes(cur)?cur:all[0]}
 document.getElementById("stats").innerHTML=[["PR-AUC (random 0.14)",t.pr_auc.toFixed(2)],["ROC-AUC",t.roc_auc.toFixed(2)],
 ["Churners caught",Math.round(t.recall*100)+"%"],["Precision",Math.round(t.precision*100)+"%"],["Top 10% riskiest hold",Math.round(t.top10pct_capture*100)+"% of churners"]]
 .map(([k,v])=>`<div class="card stat"><div class="v">${v}</div><div class="k">${k}</div></div>`).join("");predict()});
const NAMES={race:"Race",charclass:"Class",level_band:"Level band"};
fetch("/fairness").then(r=>r.json()).then(rows=>{
 const tabs=document.getElementById("tabs"),tb=document.getElementById("fair");
 function show(a){tabs.querySelectorAll("button").forEach(b=>b.classList.toggle("on",b.dataset.a===a));
  const rs=rows.filter(r=>r.attribute===a).sort((x,y)=>y.recall-x.recall);
  tb.innerHTML=`<tr><th>Group</th><th class="num">Player-months</th><th>Churners caught</th><th class="num">False-alarm rate</th></tr>`+
  rs.map(r=>`<tr><td>${r.group}</td><td class="num">${r.rows.toLocaleString()}</td>
  <td><div style="display:flex;align-items:center;gap:8px"><div class="hbar" style="width:${r.recall*70}%"></div>${Math.round(r.recall*100)}%</div></td>
  <td class="num">${Math.round(r.fpr*100)}%</td></tr>`).join("")}
 tabs.innerHTML=Object.keys(NAMES).map(a=>`<button type="button" data-a="${a}">${NAMES[a]}</button>`).join("");
 tabs.querySelectorAll("button").forEach(b=>b.onclick=()=>show(b.dataset.a));show("level_band")});
</script>
</body>
</html>"""
