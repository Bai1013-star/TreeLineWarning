// Short-term pages and saved work orders share the same dated predictions.
window.addEventListener('DOMContentLoaded',()=>{
  const windExplanation='<p class="section-note wind-offset-explanation"><strong>风偏是什么意思？</strong>风偏就是导线或树冠被风吹向对方时的横向位移。例：原净空3米、朝对方偏移0.5米，其他条件不变时剩余约2.5米。表中“受风横向位移”越大，剩余有效净空通常越小；单位是米。</p>';
  const label=(row,index)=>`第${index+1}天（${row.target_date}）`;
  const table=(headers,rows,className='')=>`<div class="forecast-table-wrap"><table class="forecast-table ${className}"><thead><tr>${headers.map(text=>`<th>${text}</th>`).join('')}</tr></thead><tbody>${rows}</tbody></table></div>`;
  const oldSummary=renderTreeSummary;
  renderTreeSummary=function(result){oldSummary(result);if(result.forecast_scope==='short'){
    const context=result.calendar_context||{},head=$('result').querySelector('.tree-summary-head');
    head.querySelector('h3').textContent=`未来${result.forecast_days}天树木预测 · ${context.selected_target_date}`;
    head.querySelector('p').textContent=`${result.input_snapshot.species} · 从${context.observation_date}起，逐日计算第1天至第${result.forecast_days}天`;
  }$('result').querySelectorAll('.distance-summary-item span').forEach(node=>{if(node.textContent==='预测风偏距离')node.textContent='受风横向位移（风偏）'});$('result').insertAdjacentHTML('beforeend',windExplanation)};
  const oldForecast=renderForecastSections;
  renderForecastSections=function(result){
    if(result.forecast_scope!=='short'){oldForecast(result);return}
    const rows=result.short_term_forecast||[];
    const growth=rows.map((row,index)=>`<tr><td>${esc(label(row,index))}</td><td>${row.predicted_height}</td><td>${row.height_growth}</td><td>${ManualDbh.display(row.predicted_diameter)}</td><td>${row.predicted_crown_width}</td></tr>`).join('');
    const distances=rows.map((row,index)=>`<tr><td>${esc(label(row,index))}</td><td>${row.predicted_horizontal_distance}</td><td>${row.predicted_vertical_distance}</td><td>${row.predicted_clearance_distance}</td><td>${row.predicted_effective_clearance}</td><td>${row.predicted_wind_offset_distance}</td><td>${row.predicted_sag}</td><td>${predictionDefectBadge(row)}</td></tr>`).join('');
    const risks=rows.map((row,index)=>`<tr><td>${esc(label(row,index))}</td><td>${hazardBadge(row)}</td><td>${row.comprehensive_risk_index}</td><td>${row.predicted_effective_clearance}</td><td>${esc(predictionDefectContent(row,result))}</td></tr>`).join('');
    $('forecastOutput').innerHTML=`<section class="daily-growth-results"><h3>一、未来${rows.length}天树木生长预测</h3>${table(['预测时间','树高(m)','累计增高(m)','胸径(cm)','冠幅(m)'],growth)}</section><section class="daily-distance-results"><h3>二、逐日树线距离预测</h3>${windExplanation}${table(['预测时间','水平(m)','垂直(m)','原始几何净空(m)','有效净空(m)','受风横向位移(m)','导线弧垂(m)','净空缺陷'],distances)}</section><section class="daily-risk-results"><h3>三、逐日四级风险预测</h3><p class="section-note">安全、一般、重大、紧急四级按本次设置的标准逐日判定。</p>${table(['预测时间','四级风险等级','风险指数R','有效净空(m)','超限依据'],risks,'daily-risk-table')}</section><section class="daily-weather-results"><h3>四、逐日天气与风险</h3>${DailyWeather.html(result)}</section>`;
  };
  const oldWorkOrder=renderWorkOrderDetailed;
  renderWorkOrderDetailed=function(result,options={}){
    if(result.forecast_scope!=='short'){oldWorkOrder(result,options);return}
    const entries=typeof treeForecastResults==='undefined'?[]:treeForecastResults.filter(entry=>entry.mode===result.prediction_mode);
    const batch=!options.single&&entries.some(entry=>entry.result===result)?entries:[{result}];
    const spans=new Map();for(const entry of batch){const span=entry.result.location_information?.tower_section||'档距1';if(!spans.has(span))spans.set(span,[]);spans.get(span).push(entry)}
    const sections=[...spans].map(([span,items])=>`<section class="work-section"><h2>${esc(span)} · ${items.length}棵树</h2>${items.map((entry,index)=>{
      const value=entry.result,input=value.input_snapshot||{},location=value.location_information||{},decision=value.risk_timeline_analysis?.maintenance_decision||{},wind=value.target_wind_contact_estimate||{},danger=value.threshold_timing?.danger_clearance||{};
      const rows=(value.short_term_forecast||[]).map((row,i)=>`<tr><td>${esc(label(row,i))}</td><td>${row.predicted_height}</td><td>${ManualDbh.display(row.predicted_diameter)}</td><td>${row.predicted_crown_width}</td><td>${row.predicted_effective_clearance}</td><td>${hazardBadge(row)}</td><td>${esc(predictionDefectContent(row,value))}</td></tr>`).join('');
      const windText=wind.status==='ALREADY_CONTACT'?'预测净空已为零':wind.beaufort_force!=null?`约${wind.beaufort_force}级风可能触线`:'接触风级证据不足';
      const dangerText=danger.days===0?'当前已达到危险净空阈值':danger.days!=null?`${danger.target_date}（${danger.days}天后）达到危险阈值`:`未来${value.forecast_days}天内未达到危险阈值`;
      return `<article class="tree-work-order"><h3>${index+1}. ${esc(location.tree_code||entry.treeName||input.tree_id)} · ${esc(input.species)}</h3><p>观测日期：${esc(value.calendar_context.observation_date)}；预测第1天至第${value.forecast_days}天。当前树高${input.height}m，胸径${ManualDbh.display(input.diameter)}cm，冠幅${input.crown_width}m，几何净空${input.clearance_distance}m。</p><p>${esc(decision.status)}：${esc(decision.recommended_deadline)}。${esc(dangerText)}；${esc(windText)}。</p>${table(['预测时间','树高(m)','胸径(cm)','冠幅(m)','有效净空(m)','四级风险','超限依据'],rows)}</article>`;
    }).join('')}</section>`).join('');
    activeWorkOrderId=String(result.work_order?.work_order_id||'');$('reportOutput').classList.add('visible');
    $('reportOutput').innerHTML=`<div class="report-toolbar"><button type="button" onclick="printWorkOrder()">打印 / 保存为 PDF</button></div><div class="work-order-title">树线逐日预测与预警工单</div><div class="work-meta"><div>生成时间：${esc(result.work_order?.generated_at)}</div><div>采用标准：${esc(result.applied_standard.profile_name)}</div><div>本次预测：未来${result.forecast_days}天</div><div>树木数量：${batch.length}</div></div>${sections}<div class="work-order-end">—— 工单结束 ——</div>`;
  };
  // The short entry uses days. Monthly inputs remain on the long-table page.
  document.querySelectorAll('.forecast-workspace > strong,.forecast-workspace > .choice-row,.forecast-workspace > .target-picker').forEach(node=>node.hidden=true);
  const target=$('integratedTargetDate');
  if(target&&$('integratedObservationDate').value){const start=new Date($('integratedObservationDate').value+'T00:00:00Z');target.value=new Date(start.getTime()+7*86400000).toISOString().slice(0,10)}
  target?.addEventListener('change',()=>{const days=Math.round((new Date(target.value+'T00:00:00Z')-new Date($('integratedObservationDate').value+'T00:00:00Z'))/86400000);if(days>=1&&days<=15)DailyWeather.setDays(days);else{target.setCustomValidity('短时预测目标请选择观测日期之后1–15天');target.reportValidity()}});
  target?.addEventListener('input',()=>target.setCustomValidity(''));
});
