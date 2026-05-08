      delete tw.pathOpsSlotCcw;
      delete tw.slotOn48;
      delete tw.slotCw48;
      delete tw.slotCcw48;
      delete tw.icaoCategoryAllowedMask;
      return;
    }
    pathOpsMigrateLegacySlotKeysInPlace(tw);
    tw.pathOpsSlotOn = pathOpsNormalizeSlotRow(tw.pathOpsSlotOn, true);
    const cwUniform = pathOpsCwRowUniformArrayForDirectionOrNull(tw);
    const cwDef = tw.pathType === 'runway_exit' ? pathOpsCwDefaultBoolForRunwayExit(tw) : pathOpsCwDefaultBoolForTaxiwayDirection(tw);
    if (cwUniform) {
      tw.pathOpsSlotCw = pathOpsNormalizeSlotRow(cwUniform, cwDef);
    } else {
      tw.pathOpsSlotCw = pathOpsNormalizeSlotRow(tw.pathOpsSlotCw, cwDef);
    }
    const ccwUniform = pathOpsCcwRowUniformArrayForDirectionOrNull(tw);
    const ccwDef = tw.pathType === 'runway_exit' ? pathOpsCcwDefaultBoolForRunwayExit(tw) : pathOpsCcwDefaultBoolForTaxiwayDirection(tw);
    if (ccwUniform) {
      tw.pathOpsSlotCcw = pathOpsNormalizeSlotRow(ccwUniform, ccwDef);
    } else {
      tw.pathOpsSlotCcw = pathOpsNormalizeSlotRow(tw.pathOpsSlotCcw, ccwDef);
    }
    if (tw.pathType === 'runway') {
      let j = 0;
      for (j = 0; j < PATH_OPS_SLOT_COUNT; j++) {
        if (tw.pathOpsSlotCw[j] && tw.pathOpsSlotCcw[j]) tw.pathOpsSlotCcw[j] = false;
      }
    }
    pathOpsCoerceTaxiAndRunwayTaxiwayMinOneCwOrCcwSlotInPlace(tw);
    let m = tw.icaoCategoryAllowedMask;
    if (typeof m !== 'number' || !isFinite(m)) m = ICAO_CAT_ALLOWED_MASK_FULL;
    tw.icaoCategoryAllowedMask = (m | 0) & ICAO_CAT_ALLOWED_MASK_FULL;
  }
  /** Path-ops slot index from touchdown anchor minutes; matches ``slot_index_from_anchor_abs_sec`` (86400-mod, floor). */
  function arrivalPathOpsSlotIndexFromFlightSimAnchor(f) {
    const sibtNorm = (f.sibtMin != null && isFinite(f.sibtMin))
      ? f.sibtMin
      : (f.timeMin != null && isFinite(f.timeMin) ? f.timeMin : 0);
    const anchorMin = Math.max(0, sibtNorm - SCHED_SIBT_MINUS_SLDT_MIN);
    let sec = Math.floor(anchorMin * 60);
    sec = ((sec % 86400) + 86400) % 86400;
    const ivSec = PATH_OPS_INTERVAL_MINUTES * 60;
    let idx = Math.floor(sec / ivSec);
    return Math.max(0, Math.min(PATH_OPS_SLOT_COUNT - 1, idx));
  }
  /** A–F only; flight 필드가 없으면 ``aircraftType`` → ``getCodeForAircraft`` (기종 정의의 RECAT 글자). 둘 다 안 되면 ``null``. */
  function flightIcaoLetterForArrivalInfra(f) {
    if (!f) return null;
    const raw = f.icaoCategory != null ? String(f.icaoCategory) : (f.icao_category != null ? String(f.icao_category) : '');
    const fromField = raw.trim().toUpperCase().charAt(0);
    if (fromField && ICAO_LETTERS_ORDER.indexOf(fromField) >= 0) return fromField;
    const typeId = f.aircraftType != null ? String(f.aircraftType).trim() : '';
    if (typeId && typeof getCodeForAircraft === 'function') {
      const fromAc = String(getCodeForAircraft(typeId) || '').trim().toUpperCase().charAt(0);
      if (fromAc && ICAO_LETTERS_ORDER.indexOf(fromAc) >= 0) return fromAc;
    }
    return null;
  }
  function resolveArrivalRunwayTaxiwayFromState(arrRunwayId) {
    if (arrRunwayId == null) return null;
    const tws = state.taxiways || [];
    for (let i = 0; i < tws.length; i++) {
      const t = tws[i];
      if (t && t.pathType === 'runway' && t.id === arrRunwayId) return t;
    }
    return null;
  }
  /** Runway path-ops row at ``slotIdx`` → single operational direction for arrival RET F2 (fallback: layout direction). */
  function arrivalEffectiveRunwayDirForSlot(rw, slotIdx) {
    if (!rw || rw.pathType !== 'runway') return null;
    if (!(slotIdx >= 0 && slotIdx < PATH_OPS_SLOT_COUNT)) return null;
    pathOpsMigrateLegacySlotKeysInPlace(rw);
    normalizePathOpsPayloadInPlaceForTaxiway(rw);
    const cw = rw.pathOpsSlotCw[slotIdx];
    const ccw = rw.pathOpsSlotCcw[slotIdx];
    if (cw && !ccw) return 'clockwise';
    if (ccw && !cw) return 'counter_clockwise';
    return getRunwayOperationalDirForArrivalRetFilter2(rw);
  }
  /** ``true`` = blocked (off or ICAO not allowed). */
  function pathOpsBlockedOpenOrIcaoAtSlot(tw, slotIdx, icaoLetter) {
    if (!tw || !icaoLetter || ICAO_LETTERS_ORDER.indexOf(icaoLetter) < 0) return true;
    if (!pathOpsEligiblePathType(tw.pathType)) return false;
    pathOpsMigrateLegacySlotKeysInPlace(tw);
    const onRow = pathOpsNormalizeSlotRow(tw.pathOpsSlotOn, true);
    if (slotIdx < 0 || slotIdx >= PATH_OPS_SLOT_COUNT) return true;
    if (!onRow[slotIdx]) return true;
    let m = tw.icaoCategoryAllowedMask;
    if (typeof m !== 'number' || !isFinite(m)) m = ICAO_CAT_ALLOWED_MASK_FULL;
    m = (m | 0) & ICAO_CAT_ALLOWED_MASK_FULL;
    const bit = ICAO_LETTERS_ORDER.indexOf(icaoLetter);
    return ((m >> bit) & 1) === 0;
  }
  function pathOpsRetCwCcwBranchOpenAtSlot(exitTw, slotIdx, runwayEffectiveDir) {
    if (!exitTw || exitTw.pathType !== 'runway_exit') return false;
    if (runwayEffectiveDir !== 'clockwise' && runwayEffectiveDir !== 'counter_clockwise') return false;
    pathOpsMigrateLegacySlotKeysInPlace(exitTw);
    const cwDef = exitTw.pathType === 'runway_exit' ? pathOpsCwDefaultBoolForRunwayExit(exitTw) : pathOpsCwDefaultBoolForTaxiwayDirection(exitTw);
    const ccwDef = exitTw.pathType === 'runway_exit' ? pathOpsCcwDefaultBoolForRunwayExit(exitTw) : pathOpsCcwDefaultBoolForTaxiwayDirection(exitTw);
    const cw = pathOpsNormalizeSlotRow(exitTw.pathOpsSlotCw, cwDef);
    const ccw = pathOpsNormalizeSlotRow(exitTw.pathOpsSlotCcw, ccwDef);
    if (slotIdx < 0 || slotIdx >= PATH_OPS_SLOT_COUNT) return false;
    return runwayEffectiveDir === 'clockwise' ? !!cw[slotIdx] : !!ccw[slotIdx];
  }
  function stripPathOpsDefaultsFromTaxiwaySerializeCopy(copy) {
    if (!copy || !pathOpsEligiblePathType(copy.pathType)) return;
    normalizePathOpsPayloadInPlaceForTaxiway(copy);
    if (pathOpsSlotsEqual(copy.pathOpsSlotOn, pathOpsDefaultSlotOnAllTrue())) delete copy.pathOpsSlotOn;
    const cwDef = copy.pathType === 'runway_exit' ? pathOpsCwDefaultBoolForRunwayExit(copy) : pathOpsCwDefaultBoolForTaxiwayDirection(copy);
    const defCw = pathOpsFilledSlotRow(cwDef);
    if (pathOpsSlotsEqual(copy.pathOpsSlotCw, defCw)) delete copy.pathOpsSlotCw;
    const ccwDef = copy.pathType === 'runway_exit' ? pathOpsCcwDefaultBoolForRunwayExit(copy) : pathOpsCcwDefaultBoolForTaxiwayDirection(copy);
    const defCcw = pathOpsFilledSlotRow(ccwDef);
    let rawDir = copy.direction != null ? String(copy.direction).trim() : '';
    if (!rawDir) rawDir = copy.pathType === 'runway' ? 'clockwise' : 'both';
    const dirN = typeof normalizeRwDirectionValue === 'function' ? normalizeRwDirectionValue(rawDir) : 'both';
    if (dirN === 'both') {
      if (copy.pathType === 'runway') {
        if (pathOpsSlotsEqual(copy.pathOpsSlotCcw, pathOpsFilledSlotRow(false))) delete copy.pathOpsSlotCcw;
      } else if (pathOpsSlotsEqual(copy.pathOpsSlotCcw, pathOpsFilledSlotRow(true))) {
        delete copy.pathOpsSlotCcw;
      }
    } else if (pathOpsSlotsEqual(copy.pathOpsSlotCcw, defCcw)) {
      delete copy.pathOpsSlotCcw;
    }
    if (copy.icaoCategoryAllowedMask === ICAO_CAT_ALLOWED_MASK_FULL) delete copy.icaoCategoryAllowedMask;
  }
  function attachNonDefaultPathOpsToSimEdge(edge, tw) {
    if (!edge || !tw) return;
    const pt = edge.pathType || '';
    if (!pathOpsEligiblePathType(pt)) return;
    const on = pathOpsNormalizeSlotRow(tw.pathOpsSlotOn, true);
    if (!pathOpsSlotsEqual(on, pathOpsDefaultSlotOnAllTrue())) edge.pathOpsSlotOn = on.slice();
    let m = tw.icaoCategoryAllowedMask;
    if (typeof m !== 'number' || !isFinite(m)) m = ICAO_CAT_ALLOWED_MASK_FULL;
    m = (m | 0) & ICAO_CAT_ALLOWED_MASK_FULL;
    if (m !== ICAO_CAT_ALLOWED_MASK_FULL) edge.icaoCategoryAllowedMask = m;
    const cwDef = pt === 'runway_exit' ? pathOpsCwDefaultBoolForRunwayExit(tw) : pathOpsCwDefaultBoolForTaxiwayDirection(tw);
    const cwNorm = pathOpsNormalizeSlotRow(tw.pathOpsSlotCw, cwDef);
    const defCw = pathOpsFilledSlotRow(cwDef);
    if (!pathOpsSlotsEqual(cwNorm, defCw)) edge.pathOpsSlotCw = cwNorm.slice();
    const ccwDef = pt === 'runway_exit' ? pathOpsCcwDefaultBoolForRunwayExit(tw) : pathOpsCcwDefaultBoolForTaxiwayDirection(tw);
    const ccwNorm = pathOpsNormalizeSlotRow(tw.pathOpsSlotCcw, ccwDef);
    let rawDir = tw.direction != null ? String(tw.direction).trim() : '';
    if (!rawDir) rawDir = tw.pathType === 'runway' ? 'clockwise' : 'both';
    const dirN = typeof normalizeRwDirectionValue === 'function' ? normalizeRwDirectionValue(rawDir) : 'both';
    let defCcwStrip;
    if (dirN === 'both' && pt === 'runway') defCcwStrip = pathOpsFilledSlotRow(false);
    else if (dirN === 'both') defCcwStrip = pathOpsFilledSlotRow(true);
    else defCcwStrip = pathOpsFilledSlotRow(ccwDef);
    if (!pathOpsSlotsEqual(ccwNorm, defCcwStrip)) edge.pathOpsSlotCcw = ccwNorm.slice();
  }
  function resetPathOpsSlotCwToDirectionDefault(tw) {
    if (!tw || !pathOpsEligiblePathType(tw.pathType)) return;
    pathOpsMigrateLegacySlotKeysInPlace(tw);
    const cwDef = tw.pathType === 'runway_exit' ? pathOpsCwDefaultBoolForRunwayExit(tw) : pathOpsCwDefaultBoolForTaxiwayDirection(tw);
    const ccwDef = tw.pathType === 'runway_exit' ? pathOpsCcwDefaultBoolForRunwayExit(tw) : pathOpsCcwDefaultBoolForTaxiwayDirection(tw);
    tw.pathOpsSlotCw = pathOpsFilledSlotRow(cwDef);
    tw.pathOpsSlotCcw = pathOpsFilledSlotRow(ccwDef);
  }
  function pathOpsSlotIntervalTooltip(idx) {
    const i = Math.max(0, Math.min(PATH_OPS_SLOT_COUNT - 1, Math.floor(Number(idx))));
    const startMin = i * PATH_OPS_INTERVAL_MINUTES;
    const hh = String(Math.floor(startMin / 60) % 24).padStart(2, '0');
    const mm = String(startMin % 60).padStart(2, '0');
    return hh + ':' + mm + '–… (' + PATH_OPS_INTERVAL_MINUTES + '분 간격)';
  }
  let pathOpsPanelDomReady = false;
  function ensurePathOpsSchedulePanelDom() {
    const wrap = document.getElementById('pathOpsScheduleWrap');
    const onHost = document.getElementById('pathOpsSlotOnGrid');
    const cwHost = document.getElementById('pathOpsSlotCwGrid');
    const ccwHost = document.getElementById('pathOpsSlotCcwGrid');
    if (!wrap || !onHost || !cwHost || !ccwHost) return;
    if (pathOpsPanelDomReady) return;
    pathOpsPanelDomReady = true;
    wrap.style.setProperty('--path-ops-n', String(PATH_OPS_SLOT_COUNT));
    function mkRow(host, cls) {
      host.innerHTML = '';
      host.classList.add('path-ops-slot-grid');
      for (let i = 0; i < PATH_OPS_SLOT_COUNT; i++) {
        const lab = document.createElement('label');
        lab.className = 'path-ops-slot-cell';
        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.checked = true;
        cb.dataset.slotIndex = String(i);
        cb.classList.add(cls);
        lab.appendChild(cb);
        lab.title = pathOpsSlotIntervalTooltip(i);
        host.appendChild(lab);
      }
    }
    mkRow(onHost, 'path-ops-on');
    mkRow(cwHost, 'path-ops-cw');
    mkRow(ccwHost, 'path-ops-ccw');
  }
  function syncPathOpsPanelFromTaxiway(tw) {
    ensurePathOpsSchedulePanelDom();
    const wrap = document.getElementById('pathOpsScheduleWrap');
    if (!wrap || !tw) return;
    normalizePathOpsPayloadInPlaceForTaxiway(tw);
    const uniCw = pathOpsCwRowUniformArrayForDirectionOrNull(tw);
    const uniCcw = pathOpsCcwRowUniformArrayForDirectionOrNull(tw);
    const onCbs = document.querySelectorAll('#pathOpsSlotOnGrid input.path-ops-on');
    const cwCbs = document.querySelectorAll('#pathOpsSlotCwGrid input.path-ops-cw');
    const ccwCbs = document.querySelectorAll('#pathOpsSlotCcwGrid input.path-ops-ccw');
    let i = 0;
    for (i = 0; i < onCbs.length && i < PATH_OPS_SLOT_COUNT; i++) {
      const el = /** @type {HTMLInputElement} */ (onCbs[i]);
      el.checked = !!tw.pathOpsSlotOn[i];
    }
    for (i = 0; i < cwCbs.length && i < PATH_OPS_SLOT_COUNT; i++) {
      const el = /** @type {HTMLInputElement} */ (cwCbs[i]);
      el.checked = !!tw.pathOpsSlotCw[i];
      el.disabled = !!uniCw;
      el.title = uniCw
        ? (pathOpsCwDefaultBoolForTaxiwayDirection(tw) ? 'Direction CW — 슬롯별 편집은 Both에서만' : 'Direction CCW — CW 허용 없음 (Both에서만 슬롯 편집)')
        : pathOpsSlotIntervalTooltip(i);
    }
    for (i = 0; i < ccwCbs.length && i < PATH_OPS_SLOT_COUNT; i++) {
      const el = /** @type {HTMLInputElement} */ (ccwCbs[i]);
      el.checked = !!tw.pathOpsSlotCcw[i];
      el.disabled = !!uniCcw;
      el.title = uniCcw
        ? (pathOpsCcwDefaultBoolForTaxiwayDirection(tw) ? 'Direction CCW — 슬롯별 편집은 Both에서만' : 'Direction CW — CCW 허용 없음 (Both에서만 슬롯 편집)')
        : pathOpsSlotIntervalTooltip(i);
    }
    pathOpsApplyCwRowDirectionLinkUi(tw);
    applyIcaoCategoriesToHost(PATH_OPS_ICAO_HOST_ID, pathOpsIcaoLettersFromMask(tw.icaoCategoryAllowedMask));
  }
  function readPathOpsPanelIntoTaxiway(tw) {
    if (!tw || !pathOpsEligiblePathType(tw.pathType)) return;
    const onCbs = document.querySelectorAll('#pathOpsSlotOnGrid input.path-ops-on');
    const cwCbs = document.querySelectorAll('#pathOpsSlotCwGrid input.path-ops-cw');
    const ccwCbs = document.querySelectorAll('#pathOpsSlotCcwGrid input.path-ops-ccw');
    const onArr = pathOpsDefaultSlotOnAllTrue();
    const cwDef = tw.pathType === 'runway_exit' ? pathOpsCwDefaultBoolForRunwayExit(tw) : pathOpsCwDefaultBoolForTaxiwayDirection(tw);
    const ccwDef = tw.pathType === 'runway_exit' ? pathOpsCcwDefaultBoolForRunwayExit(tw) : pathOpsCcwDefaultBoolForTaxiwayDirection(tw);
    const cwUniform = pathOpsCwRowUniformArrayForDirectionOrNull(tw);
    const ccwUniform = pathOpsCcwRowUniformArrayForDirectionOrNull(tw);
    const cwArr = pathOpsFilledSlotRow(cwDef);
    const ccwArr = pathOpsFilledSlotRow(ccwDef);
    let i = 0;
    for (i = 0; i < onCbs.length && i < PATH_OPS_SLOT_COUNT; i++) {
      onArr[i] = /** @type {HTMLInputElement} */ (onCbs[i]).checked;
    }
    if (!cwUniform) {
      for (i = 0; i < cwCbs.length && i < PATH_OPS_SLOT_COUNT; i++) {
        cwArr[i] = /** @type {HTMLInputElement} */ (cwCbs[i]).checked;
      }
    }
    if (!ccwUniform) {
      for (i = 0; i < ccwCbs.length && i < PATH_OPS_SLOT_COUNT; i++) {
        ccwArr[i] = /** @type {HTMLInputElement} */ (ccwCbs[i]).checked;
      }
    }
    tw.pathOpsSlotOn = onArr;
    tw.pathOpsSlotCw = cwUniform ? cwUniform.slice() : cwArr;
    tw.pathOpsSlotCcw = ccwUniform ? ccwUniform.slice() : ccwArr;
    if (tw.pathType === 'runway') {
      for (i = 0; i < PATH_OPS_SLOT_COUNT; i++) {
        if (tw.pathOpsSlotCw[i] && tw.pathOpsSlotCcw[i]) tw.pathOpsSlotCcw[i] = false;
      }
    }
    delete tw.slotOn48;
    delete tw.slotCw48;
    delete tw.slotCcw48;
    tw.icaoCategoryAllowedMask = pathOpsMaskFromIcaoLetterChecks(readIcaoCategoriesFromHost(PATH_OPS_ICAO_HOST_ID));
    syncPathOpsPanelFromTaxiway(tw);
  }
  function islandMarkerPavementResolved(m) {
    if (!m || typeof m !== 'object') return 'asphalt';
    const pv = m.pavement;
    if (pv === 'asphalt' || pv === 'cement') return pv;
    const op = m.outerPavement;
    if (op === 'taxiway') return 'cement';
    if (op === 'runway') return 'asphalt';
    return 'asphalt';
  }
  function islandMarkerPavementFillCss(m) {
    return c2dRoadWidthBandForPavement(islandMarkerPavementResolved(m));
  }
  /** Selected road-width band only: ~1% darker than theme stroke/fill. */
  const ROAD_WIDTH_SURFACE_RGB_MUL = 0.99;
  function c2dObjectSelectedGlowBlur() {
    const n = Number(_canvas2dStyle.objectSelectedGlowBlur);
    return (isFinite(n) && n >= 0) ? n : 22;
  }
  function c2dFlightSelectedRingStroke() { return _canvas2dStyle.flightSelectedRingStroke || '#facc15'; }
  function c2dFlightSelectedRingGlow() { return _canvas2dStyle.flightSelectedRingGlow || 'rgba(250, 204, 21, 0.55)'; }
  function c2dFlightSelectedRingGlowBlur() {
    const n = Number(_canvas2dStyle.flightSelectedRingGlowBlur);
    return (isFinite(n) && n >= 0) ? n : 18;
  }
  function c2dSimPreTouchdownHaloStroke() { return _canvas2dStyle.simPreTouchdownHaloStroke || 'rgba(239, 68, 68, 0.92)'; }
  function c2dSimPreTouchdownHaloFill() { return _canvas2dStyle.simPreTouchdownHaloFill || 'rgba(239, 68, 68, 0.18)'; }
  function c2dSimPreTouchdownHaloBlur() {
    const n = Number(_canvas2dStyle.simPreTouchdownHaloBlur);
    return (isFinite(n) && n >= 0) ? n : 14;
  }
  function c2dSimFlightTrailStroke() { return _canvas2dStyle.simFlightTrailStroke || 'rgba(255, 47, 146, 0.97)'; }
  function c2dSimFlightTrailStrokeEnd() { return _canvas2dStyle.simFlightTrailStrokeEnd || 'rgba(255, 47, 146, 0)'; }
  function c2dSimFlightTrailLineWidth() {
    const n = Number(_canvas2dStyle.simFlightTrailLineWidth);
    return (isFinite(n) && n > 0) ? n : 3.5;
  }
  function c2dApproachPreviewWidthM() {
    const n = Number(_canvas2dStyle.approachPreviewWidthM);
    return (isFinite(n) && n > 0) ? n : 30;
  }
  function c2dApproachPreviewStroke() {
    return _canvas2dStyle.approachPreviewStroke || 'rgba(255, 255, 255, 0.01)';
  }
  function c2dHoldingPointDiameterM() {
    const n = Number(_canvas2dStyle.holdingPointDiameterM);
    return (isFinite(n) && n > 0) ? n : 15;
  }
  function normalizeHoldingPointKind(raw) {
    return raw === 'runway_holding' ? 'runway_holding' : 'intermediate';
  }
  function pathTypeToHpKind(pathType) {
    return pathType === 'runway_exit' ? 'runway_holding' : 'intermediate';
  }
  function holdingPointKindDisplayLabel(kind) {
    return normalizeHoldingPointKind(kind) === 'runway_holding' ? 'Runway Holding Position' : 'Intermediate Holding Position';
  }
  function c2dHoldingPointFillForKind(kind) {
    const k = normalizeHoldingPointKind(kind);
    if (k === 'runway_holding') return _canvas2dStyle.holdingPointRunwayFill || 'rgba(239, 68, 68, 0.5)';
    return _canvas2dStyle.holdingPointIntermediateFill || 'rgba(249, 115, 22, 0.5)';
  }
  function c2dHoldingPointStrokeForKind(kind) {
    const k = normalizeHoldingPointKind(kind);
    if (k === 'runway_holding') return _canvas2dStyle.holdingPointRunwayStroke || 'rgba(220, 38, 38, 0.78)';
    return _canvas2dStyle.holdingPointIntermediateStroke || 'rgba(234, 88, 12, 0.75)';
  }
  function c2dHoldingPointPreviewFillForPathType(pathType) {
    const k = pathTypeToHpKind(pathType || 'taxiway');
    if (k === 'runway_holding') return _canvas2dStyle.holdingPointRunwayPreviewFill || 'rgba(239, 68, 68, 0.28)';
    return _canvas2dStyle.holdingPointIntermediatePreviewFill || 'rgba(249, 115, 22, 0.28)';
  }
  function c2dHoldingPointPreviewStrokeForPathType(pathType) {
    const k = pathTypeToHpKind(pathType || 'taxiway');
    if (k === 'runway_holding') return _canvas2dStyle.holdingPointRunwayStroke || 'rgba(220, 38, 38, 0.78)';
    return _canvas2dStyle.holdingPointIntermediateStroke || 'rgba(234, 88, 12, 0.75)';
  }
  function c2dHoldingPointMarkingYellow() {
    return _canvas2dStyle.holdingPointMarkingYellow || '#facc15';
  }
  function c2dHoldingPointMarkingLineWidthWorld() {
    const n = Number(_canvas2dStyle.holdingPointMarkingLineWidthWorld);
    return (isFinite(n) && n > 0) ? n : 0.28;
  }
  function holdingPointMarkingDoubleLineGapM(lineW) {
    const n = Number(_canvas2dStyle.holdingPointMarkingDoubleLineGapM);
    const lw = Number(lineW);
    const baseLw = (isFinite(lw) && lw > 0) ? lw : c2dHoldingPointMarkingLineWidthWorld();
    return (isFinite(n) && n > 0) ? n : Math.max(0.28, baseLw * 1.2);
  }
  function taxiwayWorldWidthMForHolding(tw) {
    if (!tw) return TAXIWAY_DEFAULT_WIDTH;
    const typ = tw.pathType || 'taxiway';
    const base = typ === 'runway' ? RUNWAY_PATH_DEFAULT_WIDTH : (typ === 'runway_exit' ? RUNWAY_EXIT_DEFAULT_WIDTH : TAXIWAY_DEFAULT_WIDTH);
    return clampTaxiwayWidthM(typ, tw.width, base);
  }
  function holdingPointBarHalfLengthMFromPathWidth(pathWidthM) {
    const w = Number(pathWidthM);
    if (isFinite(w) && w > 0) return w * 0.5;
    return Math.max(3, c2dHoldingPointDiameterM() * 0.5);
  }
  function holdingPointPerpFromTangent(ux, uy) {
    return { px: -uy, py: ux };
  }
  function distPointToSegmentSq(x, y, ax, ay, bx, by) {
    const abx = bx - ax, aby = by - ay;
    const apx = x - ax, apy = y - ay;
    const abLenSq = abx * abx + aby * aby;
    if (abLenSq < 1e-12) return apx * apx + apy * apy;
    let t = (apx * abx + apy * aby) / abLenSq;
    t = Math.max(0, Math.min(1, t));
    const qx = ax + t * abx, qy = ay + t * aby;
    const dx = x - qx, dy = y - qy;
    return dx * dx + dy * dy;
  }
  function findHoldingPointPathGeometry(hp) {
    const pt = [hp.x, hp.y];
    const wantRunway = normalizeHoldingPointKind(hp.hpKind) === 'runway_holding';
    const maxD2 = Math.pow(Math.max(CELL_SIZE * 6, 55), 2);
    let bestD2 = Infinity;
    let ux = 1, uy = 0;
    let bestTw = null;
    (state.taxiways || []).forEach(function(tw) {
      const typ = tw.pathType || 'taxiway';
      if (wantRunway) {
        if (typ !== 'runway_exit') return;
      } else {
        if (typ !== 'taxiway' && typ !== 'apron_taxiway' && typ !== 'general_queue_taxiway') return;
      }
      if (!tw.vertices || tw.vertices.length < 2) return;
      for (let i = 0; i < tw.vertices.length - 1; i++) {
        const p1 = cellToPixel(tw.vertices[i].col, tw.vertices[i].row);
        const p2 = cellToPixel(tw.vertices[i + 1].col, tw.vertices[i + 1].row);
        const near = closestPointOnSegment(p1, p2, pt);
        if (!near) continue;
        const d2 = dist2(near, pt);
        if (d2 < bestD2) {
          bestD2 = d2;
          bestTw = tw;
          const dx = p2[0] - p1[0], dy = p2[1] - p1[1];
          const len = Math.hypot(dx, dy);
          if (len > 1e-6) {
            ux = dx / len;
            uy = dy / len;
          }
        }
      }
    });
    const pathWidthM = taxiwayWorldWidthMForHolding(bestTw);
    if (bestD2 > maxD2) return { ux: 1, uy: 0, ok: false, pathWidthM, tw: bestTw };
    return { ux, uy, ok: true, pathWidthM, tw: bestTw };
  }
  function closestPointOnAnyRunwayCenterlineWorld(wx, wy) {
    const pt = [wx, wy];
    let best = null;
    let bestD2 = Infinity;
    (state.taxiways || []).forEach(function(tw) {
      if ((tw.pathType || 'taxiway') !== 'runway') return;
      if (!tw.vertices || tw.vertices.length < 2) return;
      for (let i = 0; i < tw.vertices.length - 1; i++) {
        const p1 = cellToPixel(tw.vertices[i].col, tw.vertices[i].row);
        const p2 = cellToPixel(tw.vertices[i + 1].col, tw.vertices[i + 1].row);
        const near = closestPointOnSegment(p1, p2, pt);
        if (!near) continue;
        const d2 = dist2(near, pt);
        if (d2 < bestD2) { bestD2 = d2; best = near; }
      }
    });
    return best;
  }
  function findHoldingPointPathTangent(hp) {
    const g = findHoldingPointPathGeometry(hp);
    return { ux: g.ux, uy: g.uy, ok: g.ok };
  }
  function drawHoldingPointGridMarking(ctx, cx, cy, hpKind, selected, preview) {
    const k = normalizeHoldingPointKind(hpKind);
    const g = findHoldingPointPathGeometry({ x: cx, y: cy, hpKind: hpKind });
    const { px, py } = holdingPointPerpFromTangent(g.ux, g.uy);
    const halfLen = holdingPointBarHalfLengthMFromPathWidth(g.pathWidthM);
    const pathSpanM = halfLen * 2;
    const lineW = c2dHoldingPointMarkingLineWidthWorld();
    const centerlineStroke = k === 'runway_holding' ? c2dRunwayTaxiwayCenterlineStroke() : c2dTaxiwayCenterlineStroke();
    const lineMono = layerMonoLinesOn() && !preview && !selected;
    const stroke = preview
      ? 'rgba(250, 204, 21, 0.7)'
      : (selected ? c2dObjectSelectedStroke() : (lineMono ? c2dLayerMonoLineStrokeCss() : centerlineStroke));
    const lw = preview ? Math.max(0.2, lineW * 0.92) : (selected ? lineW + 0.14 : lineW);
    const pairHalf = holdingPointMarkingDoubleLineGapM(lineW) * 0.5;
    const dashLen = Math.max(lineW * 2.2, pathSpanM * 0.13);
    const gapLen = Math.max(lineW * 1.6, pathSpanM * 0.09);
    ctx.lineCap = 'butt';
    ctx.lineJoin = 'miter';
    ctx.strokeStyle = stroke;
    ctx.lineWidth = lw;
    if (selected && !preview) {
      ctx.shadowColor = c2dObjectSelectedGlow();
      ctx.shadowBlur = c2dObjectSelectedGlowBlur();
    } else {
      ctx.shadowBlur = 0;
    }
    function strokeBarAtOffset(ofs) {
      const sx = cx - px * halfLen + g.ux * ofs;
      const sy = cy - py * halfLen + g.uy * ofs;
      const ex = cx + px * halfLen + g.ux * ofs;
      const ey = cy + py * halfLen + g.uy * ofs;
      ctx.beginPath();
      ctx.moveTo(sx, sy);
      ctx.lineTo(ex, ey);
      ctx.stroke();
    }
    if (k === 'intermediate') {
      ctx.setLineDash([dashLen, gapLen]);
      strokeBarAtOffset(0);
      ctx.setLineDash([]);
    } else {
      ctx.setLineDash([]);
      strokeBarAtOffset(-pairHalf);
      strokeBarAtOffset(pairHalf);
      const R = closestPointOnAnyRunwayCenterlineWorld(cx, cy);
      const rx = R ? R[0] : cx + g.ux * (CELL_SIZE * 40);
      const ry = R ? R[1] : cy + g.uy * (CELL_SIZE * 40);
      const midM = [cx - g.ux * pairHalf, cy - g.uy * pairHalf];
      const midP = [cx + g.ux * pairHalf, cy + g.uy * pairHalf];
      const ofsR = dist2(midM, [rx, ry]) <= dist2(midP, [rx, ry]) ? -pairHalf : pairHalf;
      const pathW = Number(g.pathWidthM);
      const toothLen = Math.max(0.75, (isFinite(pathW) && pathW > 0 ? pathW : 12) * 0.24) * 0.25;
      const toothSpacing = Math.max(0.55, pathSpanM * 0.065);
      const toothLw = Math.max(lw, lw * 1.12);
      ctx.save();
      ctx.lineWidth = toothLw;
      for (let s = -halfLen + toothSpacing * 0.5; s <= halfLen - toothSpacing * 0.25; s += toothSpacing) {
        const bx = cx + px * s + g.ux * ofsR;
        const by = cy + py * s + g.uy * ofsR;
        const mx = cx + px * s;
        const my = cy + py * s;
        const vx = rx - mx;
        const vy = ry - my;
        const signT = (g.ux * vx + g.uy * vy) >= 0 ? 1 : -1;
        ctx.beginPath();
        ctx.moveTo(bx, by);
        ctx.lineTo(bx + g.ux * signT * toothLen, by + g.uy * signT * toothLen);
        ctx.stroke();
      }
      ctx.restore();
    }
    ctx.shadowBlur = 0;
  }
  function c2dSimStandOccupiedFill() { return _canvas2dStyle.simStandOccupiedFill || 'rgba(239, 68, 68, 0.32)'; }
  function c2dSimStandOccupiedStroke() { return _canvas2dStyle.simStandOccupiedStroke || 'rgba(220, 38, 38, 0.95)'; }
  function c2dStandSafetyStroke() { return _canvas2dStyle.standSafetyStroke || 'rgba(255, 45, 110, 0.95)'; }
  function c2dPathDrawStartMarkerRadiusPx() {
    const n = Number(_canvas2dStyle.pathDrawStartMarkerRadiusPx);
    const base = (isFinite(n) && n > 0) ? n : 3.5;
    return base * LAYOUT_VERTEX_DOT_SCALE;
  }
  function c2dPathDrawStartMarkerStrokePx() {
    const n = Number(_canvas2dStyle.pathDrawStartMarkerStrokePx);
    const base = (isFinite(n) && n > 0) ? n : 1;
    return Math.max(0.5, base * LAYOUT_VERTEX_DOT_SCALE);
  }
  function c2dPathDrawStartLabelFontPx() {
    const n = Number(_canvas2dStyle.pathDrawStartLabelFontPx);
    const base = (isFinite(n) && n >= 6) ? n : 8;
    return Math.max(6, Math.round(base * LAYOUT_VERTEX_DOT_SCALE));
  }
  function c2dPathDrawStartLabelOffsetY() {
    const n = Number(_canvas2dStyle.pathDrawStartLabelOffsetY);
    const base = isFinite(n) ? n : -6;
    return base * LAYOUT_VERTEX_DOT_SCALE;
  }
  const GANTT_COLORS = {
    S_BAR: _ganttStyle.sBar || '#007aff',
    S_SERIES: _ganttStyle.sSeries || '#38bdf8',
    E_BAR: _ganttStyle.eBar || '#fb37c5',
    E_SERIES: _ganttStyle.eSeries || '#fb923c',
    CONFLICT: _ganttStyle.conflict || '#7f1d1d',
    SELECTED: _ganttStyle.selected || '#fbbf24',
  };
  const _apronAc = _layoutTier.apronAircraft || {};
  const _acScaleByCat = (_apronAc.scaleByIcaoCategory && typeof _apronAc.scaleByIcaoCategory === 'object') ? _apronAc.scaleByIcaoCategory : {};
  function apronAircraftScaleForIcao(code) {
    const c = String(code || '').toUpperCase();
    const v = Number(_acScaleByCat[c]);
    if (isFinite(v) && v > 0) return v;
    const d = Number(_acScaleByCat.default);
    return (isFinite(d) && d > 0) ? d : 1.0;
  }
  const _ac2d = _apronAc.twoD || {};
  const _acSil = (_ac2d.silhouette && typeof _ac2d.silhouette === 'object') ? _ac2d.silhouette : {};
  function apron2DGlyphFill() { return _ac2d.fillColor || '#ff2f92'; }
  function getApronAircraftDetailedSilhouettePoints() {
    const raw = _ac2d.detailedSilhouettePoints;
    if (!Array.isArray(raw) || raw.length < 3) return [];
    const out = [];
    for (let i = 0; i < raw.length; i++) {
      const row = raw[i];
      if (!Array.isArray(row) || row.length < 2) continue;
      const x = Number(row[0]);
      const y = Number(row[1]);
      if (isFinite(x) && isFinite(y)) out.push([x, y]);
    }
    return out.length >= 3 ? out : [];
  }
  const _schedAlgo = _algoTier.scheduledTimes || {};
  const SCHED_DWELL_FLOOR_MIN = (function() {
    const v = Number(_schedAlgo.dwellFloorMin);
    return (isFinite(v) && v >= 0) ? v : 20;
  })();
  /** Single S schedule: SLDT = SIBT − this many minutes; STOT = SOBT + SCHED_STOT_MINUS_SOBT_MIN. */
  const SCHED_SIBT_MINUS_SLDT_MIN = 5;
  const SCHED_STOT_MINUS_SOBT_MIN = 5;
  const RSEP_MISSING_MATRIX_SEC = (function() {
    const v = Number(_schedAlgo.rsepMissingMatrixSeparationSec);
    return (isFinite(v) && v >= 0) ? v : 90;
  })();
  const TIME_AXIS_CFG = _algoTier.timeAxis || {};
  const DOM_OPT_CFG = (_algoTier.domOptimization && typeof _algoTier.domOptimization === 'object') ? _algoTier.domOptimization : {};
  const DOM_OPT_FLIGHT_VIRT_ENABLE = DOM_OPT_CFG.flightListVirtualScroll !== false;
  const DOM_OPT_FLIGHT_VIRT_MIN = (function() {
    const v = Math.floor(Number(DOM_OPT_CFG.flightListVirtualMinRows));
    return (isFinite(v) && v >= 8) ? v : 48;
  })();
  const DOM_OPT_FLIGHT_VIRT_OVERSCAN = (function() {
    const v = Math.floor(Number(DOM_OPT_CFG.flightListVirtualOverscan));
    return (isFinite(v) && v >= 0) ? v : 8;
  })();
  const DOM_OPT_FLIGHT_VIRT_ROW_H = (function() {
    const v = Number(DOM_OPT_CFG.flightListVirtualRowHeightPx);
    return (isFinite(v) && v >= 18) ? v : 28;
  })();
  const FLIGHT_SCHED_PAGE_SIZE = (function() {
    const v = Math.floor(Number(DOM_OPT_CFG.flightSchedulePageSize));
    if (!isFinite(v) || v < 0) return 20;
    return v;
  })();
  const KPI_ROLLING_TABLE_VISIBLE_ROWS = (function() {
    const v = Math.floor(Number(DOM_OPT_CFG.kpiRollingTableVisibleRows));
    if (!isFinite(v) || v < 1) return 24;
    return v;
  })();
  function _taNum(k, def) {
    const v = Number(TIME_AXIS_CFG[k]);
    return (isFinite(v) && v >= 0) ? v : def;
  }
  const GANTT_PAD_MIN = _taNum('apronGanttPadMin', 20);
  /** Apron Gantt SIBT/SOBT resize handles snap to this many minutes. */
  const GANTT_SIBT_SOBT_HANDLE_SNAP_MIN = 5;
  const RWY_SEP_TIMELINE_PAD_MIN = _taNum('runwaySepTimelinePadMin', 10);
  const TICK_STEP_SPAN_LE60 = _taNum('tickStepWhenSpanLe60Min', 10);
  const TICK_STEP_SPAN_LE240 = _taNum('tickStepWhenSpanLe240Min', 30);
  const TICK_STEP_ELSE = _taNum('tickStepElseMin', 60);
  const MAX_TICKS_SHOWN = (function() {
    const v = Math.floor(Number(TIME_AXIS_CFG.maxTicksShown));
    return (isFinite(v) && v >= 2) ? v : 6;
  })();
  const PATH_SEARCH_CFG = _algoTier.pathSearch || {};
  const TAXIWAY_HEURISTIC_COST = (function() {
    const v = Number(PATH_SEARCH_CFG.taxiwayHeuristicCost);
    return (isFinite(v) && v > 0) ? v : 200;
  })();
  const _ix = _layoutTier.interaction || {};
  function _interactionConfigNum(k, def) {
    const v = Number(_ix[k]);
    return (isFinite(v) && v >= 0) ? v : def;
  }
  function _ixBool(k, def) {
    const v = _ix[k];
    if (typeof v === 'boolean') return v;
    if (typeof v === 'number') return v !== 0;
    if (typeof v === 'string') {
      const s = v.trim().toLowerCase();
      if (s === 'true' || s === '1' || s === 'yes' || s === 'on') return true;
      if (s === 'false' || s === '0' || s === 'no' || s === 'off') return false;
    }
    return !!def;
  }
  const LAYOUT_VERTEX_DOT_SCALE = Math.max(0.25, Math.min(1.5, _interactionConfigNum('layoutVertexDotScale', 0.7)));
  const LAYOUT_SELECTED_VERTEX_RADIUS_FACTOR = Math.max(0.25, Math.min(1.5, _interactionConfigNum('layoutSelectedVertexRadiusFactor', 0.7)));
  const GRID_VISIBLE_DEFAULT = _ixBool('showGridDefault', true);
  const IMAGE_VISIBLE_DEFAULT = _ixBool('showImageDefault', true);
  const ROAD_WIDTH_VISIBLE_DEFAULT = _ixBool('showRoadWidthDefault', true);
  const DEFAULT_LAYERS = {
    grid: GRID_VISIBLE_DEFAULT,
    image: IMAGE_VISIBLE_DEFAULT,
    pathLines: true,
    pathFill: ROAD_WIDTH_VISIBLE_DEFAULT,
    standLines: true,
    standFill: true,
    islandAreaLines: true,
    islandAreaFill: ROAD_WIDTH_VISIBLE_DEFAULT,
    buildingLines: true,
    buildingFill: true,
    textRuler: false,
    dummyFlight: false,
    junction: true
  };
  const DEFAULT_LAYER_MONO = { lines: false, fill: false, etc: false };
  const RW_EXIT_ALLOWED_DEFAULT = normalizeAllowedRunwayDirections(_dc.rwExitAllowedDefaultRaw);
  function layoutPathVertexRadiusPx(vertexSelected, pathSelected) {
    if (vertexSelected) return 6 * LAYOUT_VERTEX_DOT_SCALE * LAYOUT_SELECTED_VERTEX_RADIUS_FACTOR;
    if (pathSelected) return 5 * LAYOUT_VERTEX_DOT_SCALE * LAYOUT_SELECTED_VERTEX_RADIUS_FACTOR;
    return 4 * LAYOUT_VERTEX_DOT_SCALE;
  }
  function layoutTerminalVertexRadiusPx(vertexSelected) {
    return vertexSelected ? 5.5 * LAYOUT_VERTEX_DOT_SCALE * LAYOUT_SELECTED_VERTEX_RADIUS_FACTOR : 4 * LAYOUT_VERTEX_DOT_SCALE;
  }
  const _dragThreshPx = _interactionConfigNum('dragThresholdPx', 4);
  const DRAG_THRESH = _dragThreshPx > 0 ? Math.max(1, _dragThreshPx) : 4;
  const FREE_DRAW_STEP_CELL = Math.max(0.001, _interactionConfigNum('freeDrawStepCell', 0.05));
  const GRID_SNAP_STEP_CELL = Math.max(0.001, _interactionConfigNum('gridSnapStepCell', 0.5));
  const INSERT_VERTEX_HIT_CF = _interactionConfigNum('insertVertexHitCellFactor', 0.9);
  const CANVAS_MIN_ZOOM = Math.max(0.01, _interactionConfigNum('canvasMinZoom', 0.05));
  const CANVAS_MAX_ZOOM = Math.max(CANVAS_MIN_ZOOM, _interactionConfigNum('canvasMaxZoom', 10));
  const HIT_TERM_VTX_CF = _interactionConfigNum('hitTerminalVertexCellFactor', 0.6) * LAYOUT_VERTEX_DOT_SCALE;
  const HIT_TW_VTX_CF = _interactionConfigNum('hitTaxiwayVertexCellFactor', 0.6) * LAYOUT_VERTEX_DOT_SCALE;
  const HIT_TW_SEG_CF = _interactionConfigNum('hitTaxiwayAlongCellFactor', 0.8);
  const HIT_PBB_END_CF = _interactionConfigNum('hitPbbEndCellFactor', 0.8);
  const TRY_PBB_MAX_EDGE_CF = _interactionConfigNum('tryPlacePbbMaxEdgeCellFactor', 1.0);
  const PBB_STAND_CENTER_OFFSET_FROM_TERMINAL_WALL_M = 50;
  const PBB_NEW_CONTACT_STAND_SITE_OFFSET_M = 40;
  const FLIGHT_TOOLTIP_CF = _interactionConfigNum('flightTooltipCellFactor', 1.2);
  const FLIGHT_TOOLTIP_SCAN_MIN_MS = _interactionConfigNum('flightTooltipScanMinIntervalMs', 50);
  const TERM_CLOSE_POLY_CF = _interactionConfigNum('terminalClosePolygonCellFactor', 0.6);
  const PBB_PREVIEW_LEN_CF = _interactionConfigNum('pbbPreviewLengthCellFactor', 0.9);

  const canvas = document.getElementById('grid-canvas');
  if (canvas) {
    canvas.draggable = false;
    canvas.setAttribute('tabindex', '-1');
    canvas.style.outline = 'none';
  }
  function focusCanvasForLayoutHotkeys() {
    if (!canvas) return;
    try {
      canvas.focus({ preventScroll: true });
    } catch (e) {
      try { canvas.focus(); } catch (e2) {}
    }
  }
  const container = document.getElementById('canvas-container');
  let overlayCanvas = document.getElementById('grid-canvas-overlay');
  if (!overlayCanvas && container) {
    overlayCanvas = document.createElement('canvas');
    overlayCanvas.id = 'grid-canvas-overlay';
    container.appendChild(overlayCanvas);
  }
  let overlayCtx = (overlayCanvas && typeof overlayCanvas.getContext === 'function') ? overlayCanvas.getContext('2d') : null;
  const coordEl = document.getElementById('coord');
  const cursorPixelReadoutEl = document.getElementById('cursor-pixel-readout');
  const objectInfoEl = document.getElementById('object-info');
  const objectListEl = document.getElementById('object-list');
  const flightTooltip = document.getElementById('flight-tooltip');
  let _layoutReadoutLastCellKey = '';
  let _layoutReadoutLastPixelStr = '';
  let _layoutTooltipRafId = 0;
  let _layoutTooltipPending = null;
  const settingModeSelect = document.getElementById('settingMode');
  const layoutModeTabs = document.getElementById('layoutModeTabs');
  const panel = document.getElementById('right-panel');
  const panelToggle = document.getElementById('panel-toggle');
  const MARKER_BLAZER_COLOR_OPTIONS = ['#ff1493', '#39ff14', '#00f5ff', '#ff6600', '#ffffff'];
  const markerFlightBlazerOverlayBtn = document.createElement('button');
  const markerFlightHeadingOverlayBtn = document.createElement('button');
  const markerFlightBlazerPaletteWrap = document.createElement('div');
  markerFlightBlazerOverlayBtn.type = 'button';
  markerFlightBlazerOverlayBtn.textContent = 'Blazer: OFF';
  markerFlightBlazerOverlayBtn.setAttribute('aria-label', 'Toggle flight marker blazer');
  markerFlightBlazerOverlayBtn.style.position = 'absolute';
  markerFlightBlazerOverlayBtn.style.zIndex = '35';
  markerFlightBlazerOverlayBtn.style.display = 'none';
  markerFlightBlazerOverlayBtn.style.padding = '6px 10px';
  markerFlightBlazerOverlayBtn.style.border = '1px solid var(--ui-border-default)';
  markerFlightBlazerOverlayBtn.style.borderRadius = '6px';
  markerFlightBlazerOverlayBtn.style.background = 'var(--ui-bg-control)';
  markerFlightBlazerOverlayBtn.style.color = 'var(--ui-text-primary)';
  markerFlightBlazerOverlayBtn.style.cursor = 'pointer';
  markerFlightBlazerOverlayBtn.style.boxShadow = '0 2px 10px rgba(0,0,0,0.28)';
  markerFlightHeadingOverlayBtn.type = 'button';
  markerFlightHeadingOverlayBtn.textContent = 'Heading: FWD';
  markerFlightHeadingOverlayBtn.setAttribute('aria-label', 'Toggle flight marker heading');
  markerFlightHeadingOverlayBtn.style.position = 'absolute';
  markerFlightHeadingOverlayBtn.style.zIndex = '35';
  markerFlightHeadingOverlayBtn.style.display = 'none';
  markerFlightHeadingOverlayBtn.style.padding = '6px 10px';
  markerFlightHeadingOverlayBtn.style.border = '1px solid var(--ui-border-default)';
  markerFlightHeadingOverlayBtn.style.borderRadius = '6px';
  markerFlightHeadingOverlayBtn.style.background = 'var(--ui-bg-control)';
  markerFlightHeadingOverlayBtn.style.color = 'var(--ui-text-primary)';
  markerFlightHeadingOverlayBtn.style.cursor = 'pointer';
  markerFlightHeadingOverlayBtn.style.boxShadow = '0 2px 10px rgba(0,0,0,0.28)';
  markerFlightBlazerPaletteWrap.style.position = 'absolute';
  markerFlightBlazerPaletteWrap.style.zIndex = '35';
  markerFlightBlazerPaletteWrap.style.display = 'none';
  markerFlightBlazerPaletteWrap.style.gap = '6px';
  markerFlightBlazerPaletteWrap.style.alignItems = 'center';
  markerFlightBlazerPaletteWrap.style.padding = '4px 6px';
  markerFlightBlazerPaletteWrap.style.border = '1px solid var(--ui-border-default)';
  markerFlightBlazerPaletteWrap.style.borderRadius = '6px';
  markerFlightBlazerPaletteWrap.style.background = 'var(--ui-bg-control)';
  markerFlightBlazerPaletteWrap.style.boxShadow = '0 2px 10px rgba(0,0,0,0.28)';
  markerFlightBlazerPaletteWrap.style.pointerEvents = 'auto';
  markerFlightBlazerPaletteWrap.style.display = 'none';
  markerFlightBlazerPaletteWrap.style.flexDirection = 'row';
  function swallowBlazerOverlayPointer(ev) {
    if (!ev) return;
    ev.preventDefault();
    ev.stopPropagation();
  }
  MARKER_BLAZER_COLOR_OPTIONS.forEach(function(c) {
    const b = document.createElement('button');
    b.type = 'button';
    b.setAttribute('data-blazer-color', c);
    b.style.width = '14px';
    b.style.height = '14px';
    b.style.minWidth = '14px';
    b.style.borderRadius = '2px';
    b.style.border = '1px solid rgba(255,255,255,0.45)';
    b.style.background = c;
    b.style.cursor = 'pointer';
    b.style.padding = '0';
    b.style.margin = '0';
    markerFlightBlazerPaletteWrap.appendChild(b);
  });
  if (container) {
    container.appendChild(markerFlightBlazerOverlayBtn);
    container.appendChild(markerFlightHeadingOverlayBtn);
    container.appendChild(markerFlightBlazerPaletteWrap);
  }
  const resetViewBtn = document.getElementById('btnResetView');
  const layerPopoverBtn = document.getElementById('btnLayerPopover');
  const layerPopoverPanel = document.getElementById('layerPopoverPanel');
  const layerPopoverWrap = document.getElementById('layerPopoverWrap');
  markerFlightBlazerOverlayBtn.addEventListener('mousedown', swallowBlazerOverlayPointer);
  markerFlightBlazerOverlayBtn.addEventListener('pointerdown', swallowBlazerOverlayPointer);
  markerFlightHeadingOverlayBtn.addEventListener('mousedown', swallowBlazerOverlayPointer);
  markerFlightHeadingOverlayBtn.addEventListener('pointerdown', swallowBlazerOverlayPointer);
  markerFlightBlazerPaletteWrap.addEventListener('mousedown', swallowBlazerOverlayPointer);
  markerFlightBlazerPaletteWrap.addEventListener('pointerdown', swallowBlazerOverlayPointer);
  markerFlightBlazerOverlayBtn.addEventListener('click', function() {
    const sel = state.selectedObject;
    if (!sel || sel.type !== 'layoutMarker' || !sel.obj || sel.obj.kind !== 'flight') return;
    ensureMarkerFlightBlazerState(sel.obj);
    sel.obj.blazerEnabled = !sel.obj.blazerEnabled;
    if (sel.obj.blazerEnabled) appendMarkerFlightBlazerTrail(sel.obj);
    scheduleDraw();
    updateObjectInfo();
  });
  markerFlightBlazerPaletteWrap.addEventListener('click', function(ev) {
    const target = ev.target;
    if (!target || !target.getAttribute) return;
    const next = String(target.getAttribute('data-blazer-color') || '').trim();
    if (MARKER_BLAZER_COLOR_OPTIONS.indexOf(next) < 0) return;
    const sel = state.selectedObject;
    if (!sel || sel.type !== 'layoutMarker' || !sel.obj || sel.obj.kind !== 'flight') return;
    ensureMarkerFlightBlazerState(sel.obj);
    sel.obj.blazerColor = next;
    scheduleDraw();
    updateObjectInfo();
  });
