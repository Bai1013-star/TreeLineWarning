(function (global) {
  'use strict';

  // This file is deliberately dependency-free.  It is loaded before the local
  // Three.js/Giro3D bundle so that a broken bundle, browser runtime, or GPU
  // context can be reported without preventing the desktop shell from
  // starting.  Three.js r180 requires a WebGL2 context; WebGL1 is probed only
  // to make the diagnostic useful and is immediately released.
  const EXPECTED_THREE = '180';
  const CODES = Object.freeze({
    WEBGL_UNAVAILABLE: 'WEBGL_UNAVAILABLE',
    WEBGL_CONTEXT_CREATION_FAILED: 'WEBGL_CONTEXT_CREATION_FAILED',
    THREE_RENDERER_FAILED: 'THREE_RENDERER_FAILED',
    VIEWER_RESOURCE_MISSING: 'VIEWER_RESOURCE_MISSING',
    GPU_DRIVER_BLOCKED: 'GPU_DRIVER_BLOCKED',
    WEBVIEW2_RUNTIME_INCOMPATIBLE: 'WEBVIEW2_RUNTIME_INCOMPATIBLE',
    UNKNOWN_VIEWER_INIT_FAILURE: 'UNKNOWN_VIEWER_INIT_FAILURE',
  });

  const state = {
    environment: null,
    nativeInfo: null,
    nativePromise: null,
    failure: null,
    listeners: new Set(),
    resourceFailures: [],
    canvasRecords: typeof WeakMap === 'function' ? new WeakMap() : new Map(),
    reportSequence: 0,
  };

  const root = global || (typeof window !== 'undefined' ? window : {});

  function safeCall(fn, fallback) {
    try {
      const value = fn();
      return value === undefined ? fallback : value;
    } catch (_) {
      return fallback;
    }
  }

  function finiteNumber(value, fallback) {
    if (value === null || value === undefined || value === '') return fallback;
    const number = Number(value);
    return Number.isFinite(number) ? number : fallback;
  }

  function navigatorObject() {
    return root.navigator || (typeof navigator !== 'undefined' ? navigator : {});
  }

  function userAgent() {
    const nav = navigatorObject();
    return String(nav.userAgent || '');
  }

  function browserVersion() {
    const nav = navigatorObject();
    const brands = nav.userAgentData && Array.isArray(nav.userAgentData.brands)
      ? nav.userAgentData.brands
      : [];
    const brand = brands.find(item => /edge|edg|chrome|chromium/i.test(String(item && item.brand || '')));
    if (brand && brand.version) return String(brand.version);
    const match = userAgent().match(/(?:Edg|Edge|Chrome|Chromium|Firefox|Version)\/([\d.]+)/i);
    return match ? match[1] : null;
  }

  function getContextApi(canvas) {
    return Boolean(canvas && typeof canvas.getContext === 'function');
  }

  function hasWebglConstructor(kind) {
    const name = kind === 'webgl2' ? 'WebGL2RenderingContext' : 'WebGLRenderingContext';
    return typeof root[name] === 'function';
  }

  function contextStatus(apiAvailable, context, error, creationError) {
    let status = 'unavailable';
    if (context) status = 'created';
    else if (error) status = 'error';
    else if (apiAvailable) status = 'null';
    return {
      apiAvailable: Boolean(apiAvailable),
      contextAvailable: Boolean(context),
      supported: Boolean(context),
      contextStatus: status,
      error: error ? String(error.message || error) : null,
      exception: error ? {
        name: String(error.name || 'Error'),
        message: String(error.message || error),
        stack: boundedText(error.stack || '', 8000),
      } : null,
      statusMessage: creationError ? String(creationError.statusMessage || '') : '',
    };
  }

  function loseContext(context) {
    if (!context) return false;
    try {
      const extension = typeof context.getExtension === 'function'
        ? context.getExtension('WEBGL_lose_context')
        : null;
      if (extension && typeof extension.loseContext === 'function') {
        extension.loseContext();
        return true;
      }
    } catch (_) {
      // Context teardown is best effort.  A failed diagnostic must not mask
      // the original renderer exception.
    }
    return false;
  }

  function parameter(context, name, enumFallback, valueFallback) {
    if (!context || typeof context.getParameter !== 'function') return valueFallback;
    const enumValue = context[name] === undefined ? enumFallback : context[name];
    return safeCall(() => context.getParameter(enumValue), valueFallback);
  }

  function inspectContext(context) {
    const info = {
      renderer: null,
      vendor: null,
      maxTextureSize: null,
      contextAttributes: null,
      debugRendererInfo: false,
      debugRendererInfoError: null,
    };
    if (!context) return info;
    // Standard VENDOR/RENDERER values are useful even when the debug
    // extension is unavailable (and are intentionally not treated as a GPU
    // blocklist signal).
    info.vendor = parameter(context, 'VENDOR', 0x1f00, null);
    info.renderer = parameter(context, 'RENDERER', 0x1f01, null);
    info.maxTextureSize = finiteNumber(parameter(context, 'MAX_TEXTURE_SIZE', 0x0d33, null), null);
    info.contextAttributes = safeCall(
      () => typeof context.getContextAttributes === 'function' ? context.getContextAttributes() : null,
      null
    );
    try {
      const extension = typeof context.getExtension === 'function'
        ? context.getExtension('WEBGL_debug_renderer_info')
        : null;
      if (extension) {
        info.debugRendererInfo = true;
        const vendorEnum = extension.UNMASKED_VENDOR_WEBGL === undefined
          ? 0x9245 : extension.UNMASKED_VENDOR_WEBGL;
        const rendererEnum = extension.UNMASKED_RENDERER_WEBGL === undefined
          ? 0x9246 : extension.UNMASKED_RENDERER_WEBGL;
        info.vendor = safeCall(() => context.getParameter(vendorEnum), info.vendor);
        info.renderer = safeCall(() => context.getParameter(rendererEnum), info.renderer);
      }
    } catch (error) {
      info.debugRendererInfoError = String(error && (error.message || error) || error);
    }
    if (info.vendor != null) info.vendor = String(info.vendor);
    if (info.renderer != null) info.renderer = String(info.renderer);
    return info;
  }

  function baseCapabilities() {
    const nav = navigatorObject();
    return {
      schema: 'treeline-viewer-capabilities-v1',
      userAgent: userAgent(),
      browserVersion: browserVersion(),
      browser: {userAgent: userAgent(), version: browserVersion()},
      webview2: {
        status: state.nativeInfo && state.nativeInfo.webview2
          ? String(state.nativeInfo.webview2.status || 'UNKNOWN')
          : 'UNKNOWN',
        version: state.nativeInfo && state.nativeInfo.webview2
          ? (state.nativeInfo.webview2.version || null)
          : null,
      },
      webgl: contextStatus(false, null, null, null),
      webgl2: contextStatus(false, null, null, null),
      renderer: null,
      vendor: null,
      maxTextureSize: null,
      devicePixelRatio: finiteNumber(root.devicePixelRatio, 1),
      hardwareConcurrency: finiteNumber(nav.hardwareConcurrency, null),
      deviceMemory: finiteNumber(nav.deviceMemory, null),
      lowCapability: false,
      contextCreationError: null,
      nativeRuntime: state.nativeInfo,
    };
  }

  function isLowCapability(capabilities) {
    const caps = capabilities || {};
    const texture = finiteNumber(caps.maxTextureSize, null);
    const memory = finiteNumber(caps.deviceMemory, null);
    const cores = finiteNumber(caps.hardwareConcurrency, null);
    return Boolean(
      (texture != null && texture < 4096)
      || (memory != null && memory <= 4)
      || (cores != null && cores <= 4)
    );
  }

  function snapshot(record) {
    if (!record) return baseCapabilities();
    const caps = record.capabilities || record;
    const result = {...caps};
    result.webgl = {...(caps.webgl || {})};
    result.webgl2 = {...(caps.webgl2 || {})};
    result.browser = {...(caps.browser || {})};
    result.webview2 = {...(caps.webview2 || {})};
    result.contextAttributes = caps.contextAttributes ? {...caps.contextAttributes} : null;
    result.nativeRuntime = state.nativeInfo || caps.nativeRuntime || null;
    // Never expose a live context through a diagnostic object.  The record
    // itself remains private so a successful WebGL2 probe can be handed to
    // Three.js without making a second context.
    delete result.context;
    delete result.canvas;
    result.lowCapability = isLowCapability(result);
    return result;
  }

  function attachCreationListener(canvas, record) {
    if (!canvas || typeof canvas.addEventListener !== 'function') return;
    record.creationHandler = event => {
      record.creationError = {
        statusMessage: String(event && event.statusMessage || ''),
        message: 'webglcontextcreationerror',
      };
      if (record.webgl2) record.webgl2.statusMessage = record.creationError.statusMessage;
    };
    try { canvas.addEventListener('webglcontextcreationerror', record.creationHandler); } catch (_) {}
  }

  function detachCreationListener(canvas, record) {
    if (!canvas || !record || !record.creationHandler || typeof canvas.removeEventListener !== 'function') return;
    try { canvas.removeEventListener('webglcontextcreationerror', record.creationHandler); } catch (_) {}
    record.creationHandler = null;
  }

  function probeCanvas(canvas, attributes) {
    if (!canvas) return snapshot(null);
    const cached = state.canvasRecords.get(canvas);
    if (cached) return cached;
    const record = {canvas, context: null, creationError: null, capabilities: baseCapabilities()};
    const apiAvailable = getContextApi(canvas);
    attachCreationListener(canvas, record);
    let webgl2 = null;
    let webgl2Error = null;
    if (apiAvailable) {
      try {
        webgl2 = canvas.getContext('webgl2', attributes || {alpha: false, depth: true, stencil: false});
      } catch (error) {
        webgl2Error = error;
      }
    }
    // A canvas element can expose getContext() even when the WebGL API is not
    // implemented.  A successful context is itself proof that the API exists;
    // otherwise use the corresponding browser constructor to distinguish an
    // unavailable API from a present API that returned null/threw.
    const webgl2Api = apiAvailable && (Boolean(webgl2) || hasWebglConstructor('webgl2'));
    const webgl2Info = contextStatus(webgl2Api, webgl2, webgl2Error, record.creationError);
    record.capabilities.webgl2 = webgl2Info;
    record.contextError = webgl2Error;
    let webgl1 = null;
    let webgl1Error = null;
    if (apiAvailable && !webgl2) {
      try {
        webgl1 = canvas.getContext('webgl', attributes || {alpha: false, depth: true, stencil: false})
          || canvas.getContext('experimental-webgl', attributes || {alpha: false, depth: true, stencil: false});
      } catch (error) {
        webgl1Error = error;
      }
    }
    const webgl1Api = apiAvailable && (Boolean(webgl1) || hasWebglConstructor('webgl1'));
    const webgl1Info = contextStatus(webgl1Api, webgl1, webgl1Error, record.creationError);
    record.capabilities.webgl = webgl1 || webgl2 ? {
      ...webgl1Info,
      contextAvailable: true,
      supported: true,
      contextStatus: webgl2 ? 'webgl2' : webgl1Info.contextStatus,
    } : webgl1Info;
    if (webgl2) {
      record.context = webgl2;
      const info = inspectContext(webgl2);
      record.capabilities.renderer = info.renderer;
      record.capabilities.vendor = info.vendor;
      record.capabilities.maxTextureSize = info.maxTextureSize;
      record.capabilities.contextAttributes = info.contextAttributes;
      record.capabilities.debugRendererInfo = info.debugRendererInfo;
      record.capabilities.debugRendererInfoError = info.debugRendererInfoError;
    } else if (webgl1) {
      const info = inspectContext(webgl1);
      record.capabilities.renderer = info.renderer;
      record.capabilities.vendor = info.vendor;
      record.capabilities.maxTextureSize = info.maxTextureSize;
      record.capabilities.contextAttributes = info.contextAttributes;
      record.capabilities.debugRendererInfo = info.debugRendererInfo;
      record.capabilities.debugRendererInfoError = info.debugRendererInfoError;
      loseContext(webgl1);
    }
    record.capabilities.contextCreationError = record.creationError
      ? {...record.creationError} : null;
    record.capabilities.lowCapability = isLowCapability(record.capabilities);
    record.capabilities.nativeRuntime = state.nativeInfo;
    detachCreationListener(canvas, record);
    state.canvasRecords.set(canvas, record);
    return record;
  }

  function releaseCanvas(canvas, renderer) {
    if (renderer) {
      safeCall(() => typeof renderer.dispose === 'function' && renderer.dispose(), null);
      safeCall(() => typeof renderer.forceContextLoss === 'function' && renderer.forceContextLoss(), null);
    }
    const record = canvas ? state.canvasRecords.get(canvas) : null;
    if (record && record.context) loseContext(record.context);
    if (canvas) state.canvasRecords.delete(canvas);
    if (canvas && canvas.parentNode && typeof canvas.parentNode.removeChild === 'function') {
      try { canvas.parentNode.removeChild(canvas); } catch (_) {}
    }
  }

  function getCapabilities(canvas) {
    if (canvas) return snapshot(probeCanvas(canvas));
    if (state.environment) return snapshot(state.environment);
    const probe = safeCall(() => root.document && root.document.createElement('canvas'), null);
    if (!probe) {
      state.environment = baseCapabilities();
      return snapshot(state.environment);
    }
    const record = probeCanvas(probe, {alpha: false, depth: true, stencil: false});
    state.environment = snapshot(record);
    // The environment probe is metadata only.  Release its context so the
    // actual viewer canvas can own the one WebGL2 context Three.js renders on.
    releaseCanvas(probe);
    return snapshot(state.environment);
  }

  function rendererOptions(mode, capabilities) {
    const safe = String(mode || '').toLowerCase() === 'safe';
    const caps = capabilities || getCapabilities();
    const dpr = finiteNumber(caps && caps.devicePixelRatio, 1);
    return {
      antialias: !safe,
      alpha: false,
      depth: true,
      pixelRatio: Math.min(dpr > 0 ? dpr : 1, safe ? 1 : 1.75),
      mode: safe ? 'safe' : 'normal',
    };
  }

  function explicitGpuBlock(nativeInfo, capabilities) {
    const args = nativeInfo && typeof nativeInfo === 'object'
      ? (nativeInfo.browser_arguments || {}) : {};
    if (args.application_gpu_override === true) return true;
    const nativeStatus = nativeInfo && nativeInfo.webview2 && nativeInfo.webview2.status_message;
    const contextStatus = capabilities && capabilities.contextCreationError
      && capabilities.contextCreationError.statusMessage;
    const statusText = `${nativeStatus || ''} ${contextStatus || ''}`;
    return /(?:gpu|hardware).*(?:blocklisted|blocked by policy|disabled by policy)|(?:blocklisted|blocked by policy|disabled by policy).*(?:gpu|hardware)/i.test(statusText)
      || /webgl.*disabled by policy/i.test(statusText);
  }

  function explicitWebviewIncompatibility(nativeInfo) {
    if (!nativeInfo || typeof nativeInfo !== 'object') return false;
    const webview = nativeInfo.webview2 || {};
    if (webview.compatible === false) return true;
    return ['INCOMPATIBLE', 'UNSUPPORTED', 'MISSING', 'NOT_FOUND', 'FAILED'].includes(
      String(webview.status || '').toUpperCase()
    );
  }

  function resourceFailures() {
    const initial = Array.isArray(root.__treeViewerResourceFailures)
      ? root.__treeViewerResourceFailures : [];
    return [...state.resourceFailures, ...initial].filter((item, index, array) => {
      const key = JSON.stringify(item);
      return array.findIndex(candidate => JSON.stringify(candidate) === key) === index;
    }).slice(-20);
  }

  function classify(error, context) {
    const ctx = context || {};
    if (ctx.code && Object.values(CODES).includes(ctx.code)) return ctx.code;
    const nativeInfo = ctx.nativeRuntime || state.nativeInfo;
    if (explicitWebviewIncompatibility(nativeInfo)) return CODES.WEBVIEW2_RUNTIME_INCOMPATIBLE;
    if (ctx.resourceMissing || resourceFailures().length) return CODES.VIEWER_RESOURCE_MISSING;
    const caps = ctx.capabilities || getCapabilities();
    if (explicitGpuBlock(nativeInfo, caps)) return CODES.GPU_DRIVER_BLOCKED;
    const gl2 = caps && caps.webgl2;
    if (!gl2 || !gl2.apiAvailable) {
      // A WebGL1-only browser has a WebGL API, but it cannot satisfy the
      // WebGL2 requirement of Three.js r180.  Keep that distinct from a
      // browser with no WebGL API at all.
      if (caps && caps.webgl && caps.webgl.apiAvailable) return CODES.WEBGL_CONTEXT_CREATION_FAILED;
      return CODES.WEBGL_UNAVAILABLE;
    }
    if (!gl2.contextAvailable) return CODES.WEBGL_CONTEXT_CREATION_FAILED;
    if (ctx.phase === 'renderer' || ctx.phase === 'render' || ctx.phase === 'initial-render' || ctx.phase === 'context-lost') {
      return CODES.THREE_RENDERER_FAILED;
    }
    // A context-loss status message alone is not evidence of a blocked GPU;
    // keep it in the renderer/unknown buckets unless native diagnostics say
    // that the application explicitly disabled GPU rendering.
    if (error && /webgl|context|renderer/i.test(String(error.message || error))) {
      return CODES.THREE_RENDERER_FAILED;
    }
    return CODES.UNKNOWN_VIEWER_INIT_FAILURE;
  }

  function boundedText(value, limit) {
    const text = String(value == null ? '' : value);
    return text.length > limit ? `${text.slice(0, limit)}…` : text;
  }

  function failureObject(error, context) {
    const ctx = context || {};
    const original = error && error.viewerDiagnostic && error.viewerDiagnostic.original
      ? error.viewerDiagnostic.original : error;
    const message = boundedText(original && (original.message || original), 4000);
    const stack = boundedText(original && original.stack ? original.stack : '', 12000);
    const capabilities = snapshot(ctx.capabilities || state.environment || getCapabilities());
    const diagnostic = {
      schema: 'treeline-viewer-diagnostic-v1',
      id: `viewer-${Date.now().toString(36)}-${(++state.reportSequence).toString(36)}`,
      timestamp: new Date().toISOString(),
      code: classify(error, ctx),
      phase: boundedText(ctx.phase || 'initialization', 80),
      message,
      original: {
        name: boundedText(original && original.name || 'Error', 120),
        message,
        stack,
      },
      attempts: Array.isArray(ctx.attempts) ? ctx.attempts.slice(-2).map(attempt => ({
        mode: boundedText(attempt.mode || '', 20),
        options: attempt.options || null,
        code: attempt.code || null,
        error: attempt.error ? {
          name: boundedText(attempt.error.name || 'Error', 120),
          message: boundedText(attempt.error.message || attempt.error, 2000),
          stack: boundedText(attempt.error.stack || '', 6000),
        } : null,
        contextCreationError: attempt.contextCreationError || null,
      })) : [],
      capabilities,
      nativeRuntime: state.nativeInfo || ctx.nativeRuntime || null,
      resources: resourceFailures(),
    };
    return diagnostic;
  }

  function diagnosticText(diagnostic) {
    const d = diagnostic || state.failure || {};
    const caps = d.capabilities || {};
    const gl = caps.webgl || {};
    const gl2 = caps.webgl2 || {};
    const native = d.nativeRuntime || {};
    const webview = native.webview2 || caps.webview2 || {};
    const renderer = caps.renderer || 'unavailable';
    const vendor = caps.vendor || 'unavailable';
    const lines = [
      '3D Viewer could not start on this device.',
      `Failure code: ${d.code || CODES.UNKNOWN_VIEWER_INIT_FAILURE}`,
      `Phase: ${d.phase || 'initialization'}`,
      `User agent: ${boundedText(caps.userAgent || 'unavailable', 1000)}`,
      `Browser version: ${boundedText(caps.browserVersion || 'unavailable', 200)}`,
      `WebView2: ${webview.status || 'UNKNOWN'}${webview.version ? ` (${webview.version})` : ''}`,
      `WebGL API: ${gl.apiAvailable ? 'available' : 'unavailable'}; context: ${gl.contextAvailable ? 'created' : (gl.contextStatus || 'not created')}`,
      `WebGL2 API: ${gl2.apiAvailable ? 'available' : 'unavailable'}; context: ${gl2.contextAvailable ? 'created' : (gl2.contextStatus || 'not created')}`,
      `Renderer/GPU: ${boundedText(renderer, 500)}`,
      `Vendor: ${boundedText(vendor, 500)}`,
      `MAX_TEXTURE_SIZE: ${caps.maxTextureSize == null ? 'unavailable' : caps.maxTextureSize}`,
      `devicePixelRatio: ${caps.devicePixelRatio == null ? 'unavailable' : caps.devicePixelRatio}`,
      `Context attributes: ${caps.contextAttributes ? JSON.stringify(caps.contextAttributes) : 'unavailable'}`,
      `Three.js expected: ${EXPECTED_THREE}`,
      `Original exception: ${boundedText(d.original && d.original.message || d.message || 'unavailable', 4000)}`,
    ];
    if (caps.contextCreationError && caps.contextCreationError.statusMessage) {
      lines.push(`Context creation status: ${boundedText(caps.contextCreationError.statusMessage, 1000)}`);
    }
    if (Array.isArray(d.resources) && d.resources.length) {
      lines.push(`Resource failures: ${JSON.stringify(d.resources).slice(0, 4000)}`);
    }
    if (d.original && d.original.stack) {
      lines.push('Original stack:', boundedText(d.original.stack, 12000));
    }
    if (Array.isArray(d.attempts) && d.attempts.length) {
      lines.push('Renderer attempts:', JSON.stringify(d.attempts).slice(0, 10000));
    }
    if (native.browser_arguments) {
      lines.push(`Browser arguments: ${JSON.stringify(native.browser_arguments).slice(0, 4000)}`);
    }
    return lines.join('\n');
  }

  function showFailure(diagnostic) {
    const d = diagnostic && diagnostic.viewerDiagnostic ? diagnostic.viewerDiagnostic : diagnostic;
    if (!d) return;
    state.failure = d;
    const doc = root.document;
    if (!doc || typeof doc.getElementById !== 'function') return;
    const panel = doc.getElementById('viewer-init-failure');
    const summary = doc.getElementById('viewer-init-summary');
    const details = doc.getElementById('viewer-init-details');
    if (panel) {
      panel.hidden = false;
      if (panel.style) panel.style.display = 'block';
    }
    if (summary) summary.textContent = '3D Viewer could not start on this device.';
    if (details) {
      details.readOnly = true;
      details.value = diagnosticText(d);
    }
    const status = doc.getElementById('viewer-renderer-status');
    if (status) status.textContent = `3D Viewer 初始化失败 · ${d.code || CODES.UNKNOWN_VIEWER_INIT_FAILURE}`;
  }

  function clearFailure() {
    state.failure = null;
    const doc = root.document;
    if (!doc || typeof doc.getElementById !== 'function') return;
    const panel = doc.getElementById('viewer-init-failure');
    if (panel) {
      panel.hidden = true;
      if (panel.style) panel.style.display = 'none';
    }
  }

  async function copyDiagnostic(diagnostic) {
    const text = diagnosticText(diagnostic || state.failure);
    const nav = navigatorObject();
    try {
      if (nav.clipboard && typeof nav.clipboard.writeText === 'function') {
        await nav.clipboard.writeText(text);
        return true;
      }
    } catch (_) {}
    const doc = root.document;
    const details = doc && typeof doc.getElementById === 'function'
      ? doc.getElementById('viewer-init-details') : null;
    if (!details) return false;
    try {
      details.readOnly = true;
      details.value = text;
      if (typeof details.select === 'function') details.select();
      if (doc && typeof doc.execCommand === 'function') return Boolean(doc.execCommand('copy'));
    } catch (_) {}
    return false;
  }

  function notifyFailure(error, context) {
    const diagnostic = failureObject(error, context);
    const wrapped = error instanceof Error ? error : new Error(diagnostic.message || '3D Viewer initialization failed');
    wrapped.viewerInitFailure = true;
    wrapped.viewerDiagnostic = diagnostic;
    wrapped.code = diagnostic.code;
    wrapped.details = diagnosticText(diagnostic);
    state.failure = diagnostic;
    // Keep the original Error object and stack visible in DevTools.  The
    // bounded report below is for persistence only; it never replaces this
    // object with a generic message.
    try { root.console && root.console.error('[TreeLineViewer]', diagnostic.code, error, diagnostic.original.stack); } catch (_) {}
    showFailure(diagnostic);
    state.listeners.forEach(listener => {
      try { listener(wrapped, diagnostic); } catch (_) {}
    });
    void reportDiagnostic(diagnostic);
    return wrapped;
  }

  function subscribe(listener) {
    if (typeof listener !== 'function') return () => {};
    state.listeners.add(listener);
    return () => state.listeners.delete(listener);
  }

  async function reportDiagnostic(diagnostic) {
    const fetchImpl = root.fetch || (typeof fetch === 'function' ? fetch : null);
    if (!fetchImpl) return false;
    const payload = JSON.stringify({
      ...diagnostic,
      message: boundedText(diagnostic.message, 4000),
      original: {
        ...diagnostic.original,
        stack: boundedText(diagnostic.original && diagnostic.original.stack, 12000),
      },
    });
    const controller = typeof AbortController === 'function' ? new AbortController() : null;
    let timer = null;
    try {
      const request = fetchImpl('/api/viewer/diagnostic', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: payload,
        keepalive: true,
        ...(controller ? {signal: controller.signal} : {}),
      });
      const timeout = new Promise(resolve => {
        timer = setTimeout(() => {
          if (controller) safeCall(() => controller.abort(), null);
          resolve(null);
        }, 1500);
      });
      await Promise.race([request, timeout]);
      return true;
    } catch (_) {
      return false;
    } finally {
      if (timer != null) clearTimeout(timer);
    }
  }

  function reportSuccess(context) {
    const ctx = context || {};
    const emit = () => {
      // A failure reported while the bounded native request was in flight
      // wins over a late success snapshot.
      if (state.failure) return null;
      const success = {
        schema: 'treeline-viewer-diagnostic-v1',
        id: `viewer-${Date.now().toString(36)}-${(++state.reportSequence).toString(36)}`,
        timestamp: new Date().toISOString(),
        code: 'OK',
        success: true,
        phase: boundedText(ctx.phase || 'initialization', 80),
        rendererMode: boundedText(ctx.rendererMode || 'normal', 20),
        message: '3D Viewer initialized successfully',
        original: {name: 'None', message: '', stack: ''},
        attempts: Array.isArray(ctx.attempts) ? ctx.attempts.slice(-2) : [],
        capabilities: snapshot(ctx.capabilities || state.environment || getCapabilities()),
        nativeRuntime: state.nativeInfo || ctx.nativeRuntime || null,
        resources: resourceFailures(),
      };
      // One bounded OK snapshot is recorded for each Viewer construction. It
      // is diagnostic evidence only and cannot affect editor/project state.
      void reportDiagnostic(success);
      return success;
    };
    // Include the native WebView2 version when the request is already in
    // flight, without making renderer construction wait for it.
    if (state.nativePromise) {
      void state.nativePromise.then(emit, emit);
      return null;
    }
    return emit();
  }

  function mergeNativeInfoIntoFailure() {
    if (!state.failure || !state.nativeInfo) return;
    state.failure.nativeRuntime = state.nativeInfo;
    state.failure.capabilities = {
      ...(state.failure.capabilities || {}),
      nativeRuntime: state.nativeInfo,
      webview2: {
        ...((state.failure.capabilities && state.failure.capabilities.webview2) || {}),
        status: state.nativeInfo.webview2 && state.nativeInfo.webview2.status || 'UNKNOWN',
        version: state.nativeInfo.webview2 && state.nativeInfo.webview2.version || null,
      },
    };
    state.failure.code = classify(state.failure, {
      capabilities: state.failure.capabilities,
      phase: state.failure.phase,
      nativeRuntime: state.nativeInfo,
      resourceMissing: Array.isArray(state.failure.resources) && state.failure.resources.length > 0,
    });
    showFailure(state.failure);
    void reportDiagnostic(state.failure);
  }

  async function fetchNativeRuntime(path, deadlineMs) {
    if (state.nativePromise) return state.nativePromise;
    const fetchImpl = root.fetch || (typeof fetch === 'function' ? fetch : null);
    if (!fetchImpl) return null;
    const timeoutMs = Math.max(100, Math.min(3000, Number(deadlineMs) || 1200));
    state.nativePromise = (async () => {
      const controller = typeof AbortController === 'function' ? new AbortController() : null;
      let timer = null;
      try {
        const request = (async () => {
          const response = await fetchImpl(path || '/api/viewer/runtime', {
            headers: {Accept: 'application/json'}, cache: 'no-store',
            ...(controller ? {signal: controller.signal} : {}),
          });
          if (!response || !response.ok) return null;
          const payload = await response.json();
          return payload && typeof payload === 'object' ? payload : null;
        })();
        const timeout = new Promise((resolve, reject) => {
          timer = setTimeout(() => {
            if (controller) safeCall(() => controller.abort(), null);
            reject(new Error('viewer runtime diagnostics timed out'));
          }, timeoutMs);
        });
        const payload = await Promise.race([request, timeout]);
        if (!payload) return null;
        state.nativeInfo = payload;
        if (state.environment) {
          state.environment.webview2 = {
            status: payload.webview2 && payload.webview2.status || 'UNKNOWN',
            version: payload.webview2 && payload.webview2.version || null,
          };
          state.environment.nativeRuntime = payload;
        }
        mergeNativeInfoIntoFailure();
        return payload;
      } catch (_) {
        return null;
      } finally {
        if (timer != null) clearTimeout(timer);
      }
    })();
    return state.nativePromise;
  }

  function resourceError(event) {
    const target = event && event.target;
    const url = target && (target.src || target.href);
    if (!url || !/(?:viewer-runtime|giro3d\.bundle|tree[-_]review[-_]?(?:3d|app)|tree_viewer)\.js/i.test(String(url))) return;
    const item = {
      url: String(url),
      message: 'viewer script resource failed to load',
      stack: boundedText(event && event.error && event.error.stack || '', 6000),
    };
    state.resourceFailures.push(item);
    notifyFailure(new Error(`${item.message}: ${item.url}`), {
      phase: 'resource', resourceMissing: true, resourceUrl: item.url,
    });
  }

  function securityPolicyError(event) {
    const blocked = String(event && event.blockedURI || '');
    const source = String(event && event.sourceFile || '');
    const directive = String(event && (event.effectiveDirective || event.violatedDirective) || '');
    const viewerResource = /(?:viewer-runtime|giro3d\.bundle|tree[-_]review|tree_viewer)\.js/i.test(`${blocked} ${source}`);
    if (!viewerResource && !(/script-src/i.test(directive) && /(?:viewer|giro3d|tree)/i.test(`${blocked} ${source}`))) return;
    const item = {
      url: blocked || source || 'inline',
      message: `CSP blocked viewer resource (${directive || 'unknown directive'})`,
      stack: '',
    };
    state.resourceFailures.push(item);
    notifyFailure(new Error(item.message), {
      phase: 'resource', resourceMissing: true, resourceUrl: item.url,
    });
  }

  function initialResourceFailures() {
    const initial = Array.isArray(root.__treeViewerResourceFailures)
      ? root.__treeViewerResourceFailures : [];
    initial.forEach(item => {
      if (item && typeof item === 'object') state.resourceFailures.push(item);
    });
  }

  const api = {
    EXPECTED_THREE,
    CODES,
    getCapabilities,
    probeCanvas,
    snapshot,
    isLowCapability,
    rendererOptions,
    releaseCanvas,
    classify,
    notifyFailure,
    reportSuccess,
    subscribe,
    fetchNativeRuntime,
    resourceFailures,
    diagnosticText,
    showFailure,
    clearFailure,
    copyDiagnostic,
    getLastFailure: () => state.failure,
    getNativeRuntime: () => state.nativeInfo,
  };

  root.TreeLineViewerRuntime = api;
  initialResourceFailures();
  if (typeof root.addEventListener === 'function') {
    root.addEventListener('error', resourceError, true);
    root.addEventListener('securitypolicyviolation', securityPolicyError, true);
  }
  // Native diagnostics are useful in the panel and in the next persisted
  // report, but renderer construction never waits for this bounded request.
  void fetchNativeRuntime('/api/viewer/runtime', 1200);
})(typeof window !== 'undefined' ? window : this);
