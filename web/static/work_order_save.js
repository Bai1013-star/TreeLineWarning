// Self-contained, offline-readable work order; desktop uses native Save As.
window.WorkOrderSave = (() => {
  async function save(){
    const report=document.getElementById('reportOutput');
    if(!report?.textContent.trim())throw Error('请先完成预测，生成工单后再保存。');
    const current=report.innerHTML,originalClasses=report.className;
    const all=document.getElementById('workOrderSaveScope')?.value==='all';
    const reports=[];
    try{
      // "All" is already rendered as one concise, span-grouped work order.
      // Preserve that aggregation in the exported file instead of expanding it
      // back into one repetitive work order per tree.
      if(!all&&predictionResultsByMode[activeResultMode]){
        renderWorkOrderDetailed(predictionResultsByMode[activeResultMode],{single:true});
      }
      const clone=report.cloneNode(true);
      clone.querySelectorAll('button,.report-toolbar').forEach(node=>node.remove());
      clone.removeAttribute('id');
      reports.push(clone.outerHTML);
    }finally{report.innerHTML=current;report.className=originalClasses}
    const css=[...document.querySelectorAll('style')].map(node=>node.textContent).join('\n');
    const content='<!doctype html><html lang="zh-CN"><meta charset="UTF-8"><title>树线预警工单</title><style>'+css+'\nbody{padding:24px}.work-order{display:block!important;break-after:page}.forecast-table-wrap{overflow:auto}</style><body>'+reports.join('\n')+'</body></html>';
    return saveContent(content);
  }
  async function saveContent(content){
    const blob=new Blob([content],{type:'text/html;charset=utf-8'});
    const filename='TreeLineWarning_预警工单_'+new Date().toISOString().slice(0,10)+'.html';
    const saved=await callNativeExport('save_export_file',[filename,await blobDataUrl(blob)]);
    if(saved!==null){
      document.getElementById('workOrderSaveStatus').textContent=saved.ok?'已保存：'+saved.path:'已取消保存，工单仍保留。';
      return saved;
    }
    const url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download=filename;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
    document.getElementById('workOrderSaveStatus').textContent='已下载工单；桌面软件中会弹出另存为窗口。';
    return {ok:true,browser_download:true};
  }
  document.addEventListener('DOMContentLoaded',()=>{
    const print=document.getElementById('printReport');
    if(!print)return;
    const button=document.createElement('button');button.id='saveWorkOrder';button.type='button';button.textContent='另存为工单';button.onclick=()=>save().catch(error=>alert(error.message));print.before(button);
    const scope=document.createElement('select');scope.id='workOrderSaveScope';scope.setAttribute('aria-label','保存工单范围');scope.innerHTML='<option value="all">本次全部树木（选择顺序）</option><option value="current">当前这棵树</option>';button.before(scope);
    const status=document.createElement('span');status.id='workOrderSaveStatus';status.role='status';print.parentElement.after(status);
  });
  return {save,saveContent};
})();
