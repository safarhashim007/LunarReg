const $ = (id) => document.getElementById(id);
const labels = { transform_integrity:'Transform integrity', image_only_configuration:'Image-only configuration', input_identity:'Input identity', image_quality_gate:'Image quality gate', full_rotation_sweep:'Full rotation sweep', inlier_support:'Inlier support', spatial_coverage:'Spatial coverage', deterministic_fit_replay:'Deterministic fit replay', heldout_evaluation:'Held-out evaluation', real_subpixel_accuracy:'Real sub-pixel accuracy', output_validation:'Output validation', tmc_v9_regression:'TMC V9 regression', regression_tests:'Regression tests', no_geometry_leakage:'No geometry leakage' };
function esc(value) { return String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
async function get(path) { const r = await fetch(path); if (!r.ok) throw new Error(await r.text()); return r.json(); }
function render(data) {
  $('state').textContent = data.state.replaceAll('_',' ');
  $('state-note').textContent = data.production_ready ? 'Every declared production check passed.' : 'The console reports evidence, not guesses.';
  const p = data.rotation_progress;
  $('sweep').textContent = p ? `${p.completed ?? 0} / ${p.total ?? 12}` : 'Not started';
  $('frozen').textContent = data.frozen ? 'Frozen' : 'Not frozen';
  $('quality').textContent = data.quality_gate == null ? 'Pending' : (data.quality_gate ? 'PASS' : 'FAIL');
  $('verdict').textContent = data.production_ready ? 'PASS' : (data.readiness ? 'FAIL' : 'Pending');
  $('verdict').className = `badge ${data.production_ready ? 'pass' : data.readiness ? 'fail' : 'neutral'}`;
  const events = [
    ['Image-only processing', !!p, p ? `Rotation candidates: ${p.completed ?? 0} / ${p.total ?? 12}` : 'No progress file yet'],
    ['Transform frozen', data.frozen, data.frozen ? 'Hash-verified transform is sealed.' : 'Evaluation is blocked until freeze.'],
    ['Held-out evaluation', !!data.evaluation, data.evaluation ? `${data.evaluation.reference_pixels?.count ?? 0} controls measured.` : 'Waiting for frozen transform.'],
    ['Production readiness', !!data.readiness, data.readiness ? data.readiness.verdict : 'Waiting for all evidence.']
  ];
  $('timeline').innerHTML = events.map(([name,done,note]) => `<div class="event ${done ? 'done' : name==='Image-only processing' && p ? 'active' : ''}"><span class="dot"></span><div><strong>${esc(name)}</strong><div class="muted">${esc(note)}</div></div></div>`).join('');
  const checks = data.readiness?.checks;
  $('checks').innerHTML = checks ? Object.entries(checks).map(([key,v]) => `<div class="check"><span>${esc(labels[key] || key)}</span><span class="${v.status==='PASS'?'pass':'fail'}">${esc(v.status)}</span></div>`).join('') : '<p class="muted">No readiness report yet. Missing evidence remains pending.</p>';
  $('updated').textContent = new Date().toLocaleTimeString();
}
async function refresh() { try { const data = await get(`/api/runs/${encodeURIComponent($('run-select').value)}`); render(data); $('connection').textContent = 'Connected'; } catch (e) { $('connection').textContent = 'Unavailable'; $('state-note').textContent = e.message; } }
async function boot() { try { const data = await get('/api/runs'); $('run-select').innerHTML = data.runs.map(name => `<option value="${esc(name)}">${esc(name)}</option>`).join(''); $('connection').textContent = data.runs.length ? 'Connected' : 'No runs'; if (data.runs.length) await refresh(); } catch (e) { $('connection').textContent = 'Unavailable'; } }
$('run-select').addEventListener('change', refresh); $('refresh').addEventListener('click', refresh); boot();
