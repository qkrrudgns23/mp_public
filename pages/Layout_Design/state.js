    if (pt === 'general_queue_taxiway') return true;
    if (pt === 'runway_exit' || pt === 'runway_taxiway') return tw.queueFlow === true;
    return false;
  }
  function normalizeTaxiwayWidthInPlace(tw) {
    if (!tw || typeof tw !== 'object') return;
    const pt = tw.pathType || 'taxiway';
    const fb = pt === 'runway' ? RUNWAY_PATH_DEFAULT_WIDTH : (pt === 'runway_exit' ? RUNWAY_EXIT_DEFAULT_WIDTH : TAXIWAY_DEFAULT_WIDTH);
    if (tw.width != null) tw.width = clampTaxiwayWidthM(pt, tw.width, fb);
  }
  const RUNWAY_START_DISPLACED_THRESHOLD_DEFAULT_M = Math.max(0, Number(_runwayPathTier.startDisplacedThresholdM) || 100);
  const RUNWAY_START_BLAST_PAD_DEFAULT_M = Math.max(0, Number(_runwayPathTier.startBlastPadM) || 100);
  const RUNWAY_END_DISPLACED_THRESHOLD_DEFAULT_M = Math.max(0, Number(_runwayPathTier.endDisplacedThresholdM) || 100);
  const RUNWAY_END_BLAST_PAD_DEFAULT_M = Math.max(0, Number(_runwayPathTier.endBlastPadM) || 100);
  function c2dObjectSelectedStroke() { return _canvas2dStyle.objectSelectedStroke || 'rgba(233, 213, 255, 0.62)'; }
  function c2dObjectSelectedFill() { return _canvas2dStyle.objectSelectedFill || 'rgba(196, 181, 253, 0.28)'; }
  function c2dObjectSelectedDashStroke() { return _canvas2dStyle.objectSelectedDashStroke || 'rgba(255, 252, 255, 0.55)'; }
  function c2dObjectSelectedGlow() { return _canvas2dStyle.objectSelectedGlow || 'rgba(167, 139, 250, 0.45)'; }
  function c2dRunwayStroke() { return _canvas2dStyle.runwayStroke || 'rgba(156, 163, 175, 0.78)'; }
  function c2dRunwayFill() { return _canvas2dStyle.runwayFill || 'rgba(75, 85, 99, 0.78)'; }
  function c2dTaxiwayPavementStroke() {
    const s = _canvas2dStyle.taxiwayPavementStroke;
    return (typeof s === 'string' && s.trim()) ? s.trim() : '#827f76';
  }
  function c2dTaxiwayPavementFill() {
    const s = _canvas2dStyle.taxiwayPavementFill;
    return (typeof s === 'string' && s.trim()) ? s.trim() : '#908e82';
  }
  function c2dRunwayOutline() { return _canvas2dStyle.runwayOutline || '#cbd5e1'; }
  function c2dRunwayMarkingColor() { return _canvas2dStyle.runwayMarkingColor || '#f8fafc'; }
  function c2dRunwayThresholdColor() { return _canvas2dStyle.runwayThresholdColor || c2dRunwayMarkingColor(); }
  function c2dRunwayCenterlineColor() { return _canvas2dStyle.runwayCenterlineColor || c2dRunwayMarkingColor(); }
  function c2dRunwayTouchdownColor() { return _canvas2dStyle.runwayTouchdownColor || c2dRunwayMarkingColor(); }
  function c2dRunwayAimingPointColor() { return _canvas2dStyle.runwayAimingPointColor || c2dRunwayMarkingColor(); }
  function c2dRunwayExtensionFill() { return _canvas2dStyle.runwayExtensionFill || c2dRunwayStroke(); }
  function c2dRunwayBlastChevronColor() { return _canvas2dStyle.runwayBlastChevronColor || '#facc15'; }
  /** Strip alpha from rgba for solid road surface when showRoadWidth is on. */
  function c2dCssColorToOpaque(css) {
    const s = String(css || '').trim();
    const ra = s.match(/^rgba\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*[\d.]+\s*\)/i);
    if (ra) return 'rgb(' + ra[1] + ',' + ra[2] + ',' + ra[3] + ')';
    return s;
  }
  const C2D_COLOR_SHADE_STEP_MUL = 0.88;
  function c2dParseCssRgbTriplet(css) {
    const s = String(css || '').trim();
    let m = s.match(/^#([0-9a-f]{3})$/i);
    if (m) {
      const h = m[1];
      return [parseInt(h[0] + h[0], 16), parseInt(h[1] + h[1], 16), parseInt(h[2] + h[2], 16)];
    }
    m = s.match(/^#([0-9a-f]{6})$/i);
    if (m) {
      const h = m[1];
      return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
    }
    m = s.match(/^rgba?\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/i);
    if (m) return [Number(m[1]), Number(m[2]), Number(m[3])];
    return null;
  }
  function c2dCssColorLightenSteps(css, steps) {
    const opaque = c2dCssColorToOpaque(css);
    const t = c2dParseCssRgbTriplet(opaque);
    const n = Number(steps);
    if (!t || !(n > 0)) return opaque;
    const f = Math.pow(1 / C2D_COLOR_SHADE_STEP_MUL, n);
    const r = Math.max(0, Math.min(255, Math.round(t[0] * f)));
    const g = Math.max(0, Math.min(255, Math.round(t[1] * f)));
    const b = Math.max(0, Math.min(255, Math.round(t[2] * f)));
    return 'rgb(' + r + ',' + g + ',' + b + ')';
  }
  /** Multiply RGB channels (e.g. 0.99 ≈ 1% darker). Expects opaque-ish CSS; alpha stripped first. */
  function c2dCssColorRgbChannelScale(css, mul) {
    const opaque = c2dCssColorToOpaque(css);
    const t = c2dParseCssRgbTriplet(opaque);
    const f = Number(mul);
    if (!t || !isFinite(f)) return opaque;
    const r = Math.max(0, Math.min(255, Math.round(t[0] * f)));
    const g = Math.max(0, Math.min(255, Math.round(t[1] * f)));
    const b = Math.max(0, Math.min(255, Math.round(t[2] * f)));
    return 'rgb(' + r + ',' + g + ',' + b + ')';
  }
  /** Same rgb as layout marker kind=area fill (`drawLayoutAreaMarkers2DFloor`, 3 lighten steps). */
  function c2dRoadWidthBandSurfaceColor() {
    return c2dCssColorLightenSteps(c2dRunwayStroke(), 3);
  }
  /** Taxiway / lead-in taxiway width band: one step darker than marker area (2 lighten steps vs runway stroke). */
  function c2dRoadWidthBandTaxiwaySurfaceColor() {
    return c2dCssColorLightenSteps(c2dRunwayStroke(), 2);
  }
  /** Runway path & runway taxiway (runway_exit) width band: dark asphalt gray. */
  function c2dRoadWidthBandRunwayAsphaltColor() {
    return '#363636';
  }
  /** Layer mono: cool blue-gray (slate), not warm taupe. */
  function c2dLayerMonoLineStrokeCss() {
    return '#94a3b8';
  }
  /** Layer mono: fills match path **asphalt** width band (`c2dRoadWidthBandForPavement('asphalt')`), not cement / `c2dTaxiwayPavementFill`. */
  function c2dLayerMonoFillDarkAsphaltCss() {
    return c2dCssColorToOpaque(c2dRoadWidthBandRunwayAsphaltColor());
  }
  function c2dLayerMonoFillDarkAsphaltRgba(a) {
    const t = c2dParseCssRgbTriplet(c2dLayerMonoFillDarkAsphaltCss());
    const al = Number(a);
    if (!t || !isFinite(al)) return c2dLayerMonoFillDarkAsphaltCss();
    return 'rgba(' + t[0] + ',' + t[1] + ',' + t[2] + ',' + Math.max(0, Math.min(1, al)) + ')';
  }
  const C2D_LAYER_MONO_ETC_WHITE = '#f8fafc';
  function pathPavementDefaultForPathType(pathType) {
    const pt = pathType || 'taxiway';
    if (pt === 'runway' || pt === 'runway_exit') return 'asphalt';
    return 'cement';
  }
  function pathPavementResolvedForTaxiway(tw) {
    if (!tw || typeof tw !== 'object') return 'cement';
    const v = tw.pavement;
    if (v === 'asphalt' || v === 'cement') return v;
    return pathPavementDefaultForPathType(tw.pathType);
  }
  function c2dRoadWidthBandForPavement(pavement) {
    return pavement === 'cement' ? c2dRoadWidthBandTaxiwaySurfaceColor() : c2dRoadWidthBandRunwayAsphaltColor();
  }
  function normalizePathPavementInPlace(tw) {
    if (!tw || typeof tw !== 'object') return;
    const v = tw.pavement;
    if (v === 'asphalt' || v === 'cement') return;
    tw.pavement = pathPavementDefaultForPathType(tw.pathType);
  }
  const ICAO_CAT_ALLOWED_MASK_FULL = 0x3f;
  /** DOM id: same `icao-multi-checks` pattern as Contact Stand. */
  const PATH_OPS_ICAO_HOST_ID = 'taxiwayPathOpsIcaoCategories';
  function pathOpsIcaoLettersFromMask(mask) {
    let m = mask;
    if (typeof m !== 'number' || !isFinite(m)) m = ICAO_CAT_ALLOWED_MASK_FULL;
    m = (m | 0) & ICAO_CAT_ALLOWED_MASK_FULL;
    if (m === 0) return ICAO_LETTERS_ORDER.slice();
    const out = [];
    ICAO_LETTERS_ORDER.forEach(function(L, bit) {
      if (((m >> bit) & 1) === 1) out.push(L);
    });
    return out;
  }
  function pathOpsMaskFromIcaoLetterChecks(letters) {
    const arr = normalizeAllowedIcaoCategories(letters);
    if (!arr.length) return ICAO_CAT_ALLOWED_MASK_FULL;
    let mask = 0;
    arr.forEach(function(L) {
      const ix = ICAO_LETTERS_ORDER.indexOf(L);
      if (ix >= 0) mask |= 1 << ix;
    });
    return (mask | 0) & ICAO_CAT_ALLOWED_MASK_FULL;
  }
  function pathOpsEligiblePathType(pt) {
    return pt === 'runway' || pt === 'runway_taxiway' || pt === 'runway_exit' ||
      pt === 'taxiway' || pt === 'apron_taxiway' || pt === 'general_queue_taxiway';
  }
  /** Copy legacy slotOn48 / slotCw48 / slotCcw48 into pathOps* then remove legacy keys. */
  function pathOpsMigrateLegacySlotKeysInPlace(tw) {
    if (!tw || typeof tw !== 'object') return;
    if (!Array.isArray(tw.pathOpsSlotOn) && Array.isArray(tw.slotOn48)) tw.pathOpsSlotOn = tw.slotOn48;
    if (!Array.isArray(tw.pathOpsSlotCw) && Array.isArray(tw.slotCw48)) tw.pathOpsSlotCw = tw.slotCw48;
    if (!Array.isArray(tw.pathOpsSlotCcw) && Array.isArray(tw.slotCcw48)) tw.pathOpsSlotCcw = tw.slotCcw48;
    delete tw.slotOn48;
    delete tw.slotCw48;
    delete tw.slotCcw48;
  }
  function pathOpsDefaultSlotOnAllTrue() {
    const a = [];
    for (let i = 0; i < PATH_OPS_SLOT_COUNT; i++) a.push(true);
    return a;
  }
  function pathOpsFilledSlotRow(boolVal) {
    const a = [];
    for (let i = 0; i < PATH_OPS_SLOT_COUNT; i++) a.push(!!boolVal);
    return a;
  }
  function pathOpsNormalizeSlotRow(raw, fillMissingWith) {
    const out = [];
    const fm = !!fillMissingWith;
    const src = Array.isArray(raw) ? raw : [];
    for (let i = 0; i < PATH_OPS_SLOT_COUNT; i++) {
      out.push(i < src.length ? !!src[i] : fm);
    }
    return out;
  }
  function pathOpsSlotsEqual(a, b) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== PATH_OPS_SLOT_COUNT || b.length !== PATH_OPS_SLOT_COUNT) {
      return false;
    }
    for (let i = 0; i < PATH_OPS_SLOT_COUNT; i++) {
      if (!!a[i] !== !!b[i]) return false;
    }
    return true;
  }
  function pathOpsCwDefaultBoolForTaxiwayDirection(tw) {
    if (!tw) return true;
    if (tw.pathType === 'runway') {
      const d = String(tw.direction || 'clockwise').trim();
      return d !== 'counter_clockwise';
    }
    const d2 = String(tw.direction || 'both').trim();
    return d2 !== 'counter_clockwise';
  }
  /** CCW Infra column default: Both → permissive true (matches legacy omission). */
  function pathOpsCcwDefaultBoolForTaxiwayDirection(tw) {
    if (!tw) return true;
    if (tw.pathType === 'runway') {
      const d = String(tw.direction || 'clockwise').trim();
      return d === 'counter_clockwise';
    }
    const d2 = String(tw.direction || 'both').trim();
    if (d2 === 'counter_clockwise') return true;
    if (d2 === 'clockwise') return false;
    return true;
  }
  /**
   * runway_exit: ``allowedRwDirections``가 있으면 인프라 CW/CCW 기본값을 그 목록과 맞춤.
   * 속성이 없으면 F2 레거시(양방향 허용)와 맞추어 CW·CCW 브랜치 기본을 모두 연다
   * (폴리선 ``direction``이 CW만으로 CCW 착지 브랜치를 닫지 않음).
   */
  function pathOpsCwDefaultBoolForRunwayExit(tw) {
    if (!tw || tw.pathType !== 'runway_exit') return pathOpsCwDefaultBoolForTaxiwayDirection(tw);
    if (Object.prototype.hasOwnProperty.call(tw, 'allowedRwDirections')) {
      const arr = normalizeAllowedRunwayDirections(tw.allowedRwDirections);
      if (arr.length === 0) return false;
      return arr.indexOf('clockwise') >= 0;
    }
    return true;
  }
  function pathOpsCcwDefaultBoolForRunwayExit(tw) {
    if (!tw || tw.pathType !== 'runway_exit') return pathOpsCwDefaultBoolForTaxiwayDirection(tw);
    if (Object.prototype.hasOwnProperty.call(tw, 'allowedRwDirections')) {
      const arr = normalizeAllowedRunwayDirections(tw.allowedRwDirections);
      if (arr.length === 0) return false;
      return arr.indexOf('counter_clockwise') >= 0;
    }
    return true;
  }
  /**
   * Direction Mode가 CW 또는 CCW일 때만: CW 슬롯 행은 전 구간 동일 값(방향과 연동).
   * Both일 때만 슬롯별 편집 허용.
   */
  function pathOpsCwRowUniformArrayForDirectionOrNull(tw) {
    if (!tw || !pathOpsEligiblePathType(tw.pathType)) return null;
    if (tw.pathType === 'runway_exit') return null;
    let raw = tw.direction != null ? String(tw.direction).trim() : '';
    if (!raw) raw = tw.pathType === 'runway' ? 'clockwise' : 'both';
    const d = typeof normalizeRwDirectionValue === 'function' ? normalizeRwDirectionValue(raw) : 'both';
    if (d === 'both') return null;
    return pathOpsFilledSlotRow(pathOpsCwDefaultBoolForTaxiwayDirection(tw));
  }
  function pathOpsCcwRowUniformArrayForDirectionOrNull(tw) {
    if (!tw || !pathOpsEligiblePathType(tw.pathType)) return null;
    if (tw.pathType === 'runway_exit') return null;
    let raw = tw.direction != null ? String(tw.direction).trim() : '';
    if (!raw) raw = tw.pathType === 'runway' ? 'clockwise' : 'both';
    const d = typeof normalizeRwDirectionValue === 'function' ? normalizeRwDirectionValue(raw) : 'both';
    if (d === 'both') return null;
    return pathOpsFilledSlotRow(pathOpsCcwDefaultBoolForTaxiwayDirection(tw));
  }
  /** Taxiway / Runway taxiway: each slot must allow at least one of CW or CCW (Python path-ops aligns to one branch). */
  function pathOpsCoerceTaxiAndRunwayTaxiwayMinOneCwOrCcwSlotInPlace(tw) {
    if (!tw || (tw.pathType !== 'taxiway' && tw.pathType !== 'runway_taxiway')) return;
    let j = 0;
    for (j = 0; j < PATH_OPS_SLOT_COUNT; j++) {
      if (!tw.pathOpsSlotCw[j] && !tw.pathOpsSlotCcw[j]) {
        if (pathOpsCwDefaultBoolForTaxiwayDirection(tw)) tw.pathOpsSlotCw[j] = true;
        else tw.pathOpsSlotCcw[j] = true;
      }
    }
  }
  function pathOpsApplyCwRowDirectionLinkUi(tw) {
    const linked = !!(tw && pathOpsCwRowUniformArrayForDirectionOrNull(tw));
    [
      ['pathOpsSlotCwGrid', 'path-ops-cw-row--direction-linked', 'path-ops-cw-grid--locked'],
      ['pathOpsSlotCcwGrid', 'path-ops-ccw-row--direction-linked', 'path-ops-ccw-grid--locked']
    ].forEach(function(spec) {
      const host = document.getElementById(spec[0]);
      const row = host && typeof host.closest === 'function' ? host.closest('.path-ops-slot-row') : null;
      if (!row || !host) return;
      row.classList.toggle(spec[1], linked);
      host.classList.toggle(spec[2], linked);
    });
  }
  function normalizePathOpsPayloadInPlaceForTaxiway(tw) {
    if (!tw || typeof tw !== 'object') return;
    if (!pathOpsEligiblePathType(tw.pathType)) {
      delete tw.pathOpsSlotOn;
      delete tw.pathOpsSlotCw;
