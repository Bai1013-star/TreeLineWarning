const {chromium}=require(process.env.PLAYWRIGHT_MODULE);
const fs=require('node:fs'),assert=require('node:assert/strict');
(async()=>{
  const urls=JSON.parse(fs.readFileSync(process.env.REAL_SERVERS,'utf8'));
  const out=process.env.REAL_OUTPUT;
  const browser=await chromium.launch({channel:'msedge',headless:true});
  const page=await browser.newPage({viewport:{width:1500,height:950}});
  await page.addInitScript(()=>{
    const state={username:'000000',password:'000000',first:true,authenticated:false,profile:null};
    const api={
      login:async(username,password)=>{
        if(username!==state.username||password!==state.password)return{ok:false,error:'用户名或密码错误'};
        state.authenticated=true;
        return{ok:true,authenticated:true,username:state.username,must_change_password:state.first,first_login:state.first,profile_setup_required:!state.profile,real_name:state.profile?.real_name,avatar_data_url:state.profile?.avatar};
      },
      change_password:async(current,next,confirm,username,real_name,phone,avatar)=>{
        if(!state.authenticated)return{ok:false,error:'请先登录'};
        if(current!==state.password)return{ok:false,error:'当前密码不正确'};
        if(next!==confirm)return{ok:false,error:'两次输入的新密码不一致'};
        if(next==='000000')return{ok:false,error:'新密码不能继续使用默认密码 000000'};
        state.username=username||state.username;state.password=next;state.first=false;
        if(!state.profile)state.profile={real_name,phone,avatar};
        return{ok:true,authenticated:true,username:state.username,must_change_password:false,first_login:false,real_name:state.profile.real_name,avatar_data_url:state.profile.avatar};
      },
      reset_password:async(username,real_name,phone,avatar,next,confirm)=>{
        if(username!==state.username||real_name!==state.profile?.real_name||phone!==state.profile?.phone||avatar!==state.profile?.avatar)return{ok:false,error:'身份资料不一致'};
        if(next!==confirm)return{ok:false,error:'两次输入的新密码不一致'};
        state.password=next;state.authenticated=false;return{ok:true,authenticated:false,message:'密码已重置，请使用新密码登录'};
      },
      logout:async()=>{state.authenticated=false;return{ok:true,authenticated:false}},
    };
    window.pywebview={api:new Proxy(api,{get(target,key){if(key==='then')return undefined;return key in target?target[key]:async()=>({ok:true})}})};
  });
  await page.goto(urls.desktop);
  assert.equal(await page.locator('#loginGate').isVisible(),true);
  assert.equal(await page.locator('#loginUsername').inputValue(),'000000');
  assert.equal(await page.locator('#loginPassword').inputValue(),'000000');
  await page.locator('#loginSubmit').click();
  await page.locator('#newPassword').waitFor({state:'visible'});
  assert.match(await page.locator('#loginTitle').innerText(),/首次登录/);
  await page.locator('#newUsername').fill('巡检员01');
  await page.locator('#newPassword').fill('secure-2026');
  await page.locator('#confirmPassword').fill('secure-2026');
  await page.locator('#realName').fill('张三');
  await page.locator('#phoneNumber').fill('13800138000');
  await page.locator('#avatarFile').setInputFiles({name:'avatar.png',mimeType:'image/png',buffer:Buffer.from('test-avatar')});
  await page.locator('#loginSubmit').click();
  await page.locator('#accountChip').waitFor({state:'visible'});
  assert.equal(await page.locator('#accountName').innerText(),'张三（巡检员01）');
  await page.locator('#changePasswordButton').click();
  await page.locator('#loginPassword').fill('secure-2026');
  await page.locator('#newPassword').fill('secure-2026-next');
  await page.locator('#confirmPassword').fill('secure-2026-next');
  await page.locator('#loginSubmit').click();
  await page.locator('#accountChip').waitFor({state:'visible'});
  await page.locator('#logoutButton').click();
  await page.locator('#loginGate').waitFor({state:'visible'});
  await page.locator('#loginUsername').fill('巡检员01');
  await page.locator('#loginPassword').fill('secure-2026-next');
  await page.locator('#loginSubmit').click();
  await page.locator('#accountChip').waitFor({state:'visible'});
  assert.equal(await page.locator('#accountName').innerText(),'张三（巡检员01）');
  await page.locator('#logoutButton').click();
  await page.locator('#forgotPasswordButton').click();
  await page.locator('#loginUsername').fill('巡检员01');
  await page.locator('#realName').fill('张三');
  await page.locator('#phoneNumber').fill('13800138000');
  await page.locator('#avatarFile').setInputFiles({name:'avatar.png',mimeType:'image/png',buffer:Buffer.from('test-avatar')});
  await page.locator('#newPassword').fill('recovered-2026');
  await page.locator('#confirmPassword').fill('recovered-2026');
  await page.locator('#loginSubmit').click();
  await page.locator('#loginError').filter({hasText:'密码已重置'}).waitFor({state:'visible'});
  assert.match(await page.locator('#loginError').innerText(),/密码已重置/);
  await page.locator('#loginUsername').fill('巡检员01');
  await page.locator('#loginPassword').fill('recovered-2026');
  await page.locator('#loginSubmit').click();
  await page.locator('#accountChip').waitFor({state:'visible'});
  if(out)await page.screenshot({path:out+'/login_20261011.png',fullPage:true});
  await browser.close();
  console.log(JSON.stringify({default_login:true,forced_change:true,identity_profile:true,repeat_change:true,username_chip:true,logout_relogin:true,forgot_password_reset:true}));
})().catch(error=>{console.error(error);process.exitCode=1});
