/**
 * Pure P3A viewer utilities shared by the POC page and the automated tests.
 *
 * This file has NO DOM access and NO startup side effects, so the test harness
 * can execute it directly with Node as well as the browser page loading it.  It
 * attaches itself to the global object (window in the browser, globalThis in
 * Node).
 */
(function (global) {
	"use strict";

	// Canonical fallback for the point-cloud load timeout (ms).
	var DEFAULT_LOAD_TIMEOUT_MS = 30000;

	/**
	 * Resolve the point-cloud load timeout (ms) from a configured value.
	 *
	 * Only a finite, strictly positive JavaScript Number is accepted; every
	 * other value (negative, zero, NaN, +/-Infinity, strings, null, booleans,
	 * objects, arrays, undefined) falls back to DEFAULT_LOAD_TIMEOUT_MS so an
	 * invalid configuration can never crash or distort viewer startup.
	 */
	function resolveLoadTimeoutMs(value) {
		if (typeof value === "number" && isFinite(value) && value > 0) {
			return value;
		}
		return DEFAULT_LOAD_TIMEOUT_MS;
	}

	/**
	 * Decide whether the viewer utility global is usable (P3A-010).
	 *
	 * Returns ``{ok: true, util: value}`` when ``value`` is a present object
	 * that exposes ``resolveLoadTimeoutMs`` as a function; otherwise
	 * ``{ok: false}`` so the startup guard can fail visibly instead of
	 * dereferencing a missing, partial, or corrupt helper.
	 */
	function decideViewerStartup(value) {
		if (
			value &&
			typeof value === "object" &&
			typeof value.resolveLoadTimeoutMs === "function"
		) {
			return { ok: true, util: value };
		}
		return { ok: false };
	}

	global.p3aViewerUtil = {
		DEFAULT_LOAD_TIMEOUT_MS: DEFAULT_LOAD_TIMEOUT_MS,
		resolveLoadTimeoutMs: resolveLoadTimeoutMs,
	};

	// Exposed separately so the startup guard can validate the utility object
	// without depending on that object itself having loaded.
	global.p3aDecideViewerStartup = decideViewerStartup;
})(typeof window !== "undefined" ? window : globalThis);
