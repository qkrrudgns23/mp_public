    }).join('||');
  }
  function stripPathGraphCacheJunctionsNearTaxiwayWorld(tw) {
    const g = state.pathGraphCache;
    if (!g || g.__junctionStale || !tw) return;
    const pts = typeof getOrderedPoints === 'function' ? getOrderedPoints(tw) : null;
    if (!pts || pts.length < 2) return;
    const mergeR = (typeof PATH_JUNCTION_MERGE_RADIUS_PX === 'number' && isFinite(PATH_JUNCTION_MERGE_RADIUS_PX)) ? PATH_JUNCTION_MERGE_RADIUS_PX : 8;
    const tol = Math.max(mergeR * 2.2, 12);
    const tol2 = tol * tol;
    function distPointToSegSq(p, a, b) {
      const pr = projectOnSegment(a, b, p);
      const dpx = p[0] - pr.p[0], dpy = p[1] - pr.p[1];
      return dpx * dpx + dpy * dpy;
    }
    function pointNearDeletedTw(p) {
      if (!p || !Array.isArray(p) || p.length < 2) return false;
      if (!isFinite(p[0]) || !isFinite(p[1])) return false;
      for (let seg = 0; seg < pts.length - 1; seg++) {
        const a = pts[seg], b = pts[seg + 1];
        if (!a || !b || !isFinite(a[0]) || !isFinite(b[0])) continue;
        if (distPointToSegSq(p, a, b) <= tol2) return true;
      }
      return false;
    }
    function filt(arr) {
      if (!Array.isArray(arr)) return arr;
      return arr.filter(function(pt) { return !pointNearDeletedTw(pt); });
    }
    if (Array.isArray(g.validJunctions)) g.validJunctions = filt(g.validJunctions);
    if (Array.isArray(g.connectedJunctions)) g.connectedJunctions = filt(g.connectedJunctions);
    if (Array.isArray(g.junctions)) g.junctions = filt(g.junctions);
    if (Array.isArray(g.disconnectedValidJunctions)) g.disconnectedValidJunctions = filt(g.disconnectedValidJunctions);
  }
  function markPathGraphJunctionStaleShellAfterLayoutEdit() {
    cancelPathGraphRebuildTimer();
    state.pathGraphCache = {
      __junctionStale: true,
      validJunctions: [],
      connectedJunctions: [],
      disconnectedValidJunctions: [],
      junctions: [],
      nodes: [],
      edges: [],
      adj: [],
      edgeMap: {},
      runwayNodeIndicesById: {},
      standIdToNodeIndex: {}
    };
    state.pathGraphCacheValid = true;
    state.pathGraphCacheSig = computeTaxiwaysGraphSig();
    state.pathGraphCacheDirty = true;
    state.pathGraphInvalidatedAtMs = Date.now();
  }
  function graphSigParseRecords(sig) {
    const m = {};
    if (!sig || typeof sig !== 'string') return m;
    const chunks = sig.split('||');
    for (let i = 0; i < chunks.length; i++) {
      const rec = chunks[i];
      if (!rec) continue;
      const pipe = rec.indexOf('|');
      const id = pipe >= 0 ? rec.slice(0, pipe) : rec;
      if (id) m[id] = rec;
    }
    return m;
  }
  function graphSigTaxiwayDiff(oldSig, newSig) {
    const o = graphSigParseRecords(oldSig);
    const n = graphSigParseRecords(newSig);
    const removed = [];
    const changed = [];
    Object.keys(o).forEach(function(id) {
      if (!(id in n)) removed.push(id);
    });
    Object.keys(n).forEach(function(id) {
      if (!(id in o)) changed.push(id);
      else if (o[id] !== n[id]) changed.push(id);
    });
    return { removed: removed, changed: changed };
  }
  function cloneFlightsWithoutPathPolylineCache(flights) {
    return (flights || []).map(function(f) {
      const raw = JSON.parse(JSON.stringify(f));
      delete raw.cachedArrPathPts;
      delete raw.cachedDepPathPts;
      delete raw._pathPolylineCacheRev;
      delete raw._pathPolylineArrRetKey;
      return raw;
    });
  }
  function markGlobalUpdateStale() {
    state.globalUpdateFresh = false;
    state.simPlaying = false;
    state.simSliderScrubbing = false;
    if (typeof ensureSimLoop === 'function') ensureSimLoop._playKick = false;
    bumpPathPolylineCacheRev();
    state.rwySepPanelDirty = true;
    bumpRwySepSnapshotStaleGen();
    if (typeof clearAllFlightTimelines === 'function') clearAllFlightTimelines({ keepDesResultTimelines: true });
    if (typeof applySimPlaybackBarDomVisibility === 'function') applySimPlaybackBarDomVisibility();
    if (typeof syncProSimButtonFromDesignerPageState === 'function') syncProSimButtonFromDesignerPageState();
  }
  function markGlobalUpdateFresh() {
    state.globalUpdateFresh = true;
    if (typeof applySimPlaybackBarDomVisibility === 'function') applySimPlaybackBarDomVisibility();
    if (typeof syncProSimButtonFromDesignerPageState === 'function') syncProSimButtonFromDesignerPageState();
  }
  function markProSimSyncStaleFromSchedule() {
    state.globalUpdateFresh = false;
    state.simPlaying = false;
    state.simSliderScrubbing = false;
    if (typeof ensureSimLoop === 'function') ensureSimLoop._playKick = false;
    if (typeof applySimPlaybackBarDomVisibility === 'function') applySimPlaybackBarDomVisibility();
    if (typeof syncProSimButtonFromDesignerPageState === 'function') syncProSimButtonFromDesignerPageState();
  }
  function markDesignerPageUpdateStale() {
    state.designerPageUpdateFresh = false;
    const dot = document.getElementById('designerPageUpdateSyncDot');
    if (dot) {
      dot.classList.remove('fresh');
      dot.classList.add('stale');
      dot.setAttribute('title', '레이아웃/객체 변경됨 — Layout Update를 눌러 경로 그래프·뷰를 동기화하세요');
    }
    const layoutUpdBtn = document.getElementById('btnDesignerPageUpdate');
    if (layoutUpdBtn) {
      layoutUpdBtn.classList.add('layout-update-stale');
      layoutUpdBtn.classList.remove('layout-update-fresh');
    }
    if (typeof syncProSimButtonFromDesignerPageState === 'function') syncProSimButtonFromDesignerPageState();
  }
  function markDesignerPageUpdateFresh() {
    state.designerPageUpdateFresh = true;
    const dot = document.getElementById('designerPageUpdateSyncDot');
    if (dot) {
      dot.classList.remove('stale');
      dot.classList.add('fresh');
      dot.setAttribute('title', 'Layout Update 기준으로 경로·뷰가 최신입니다');
    }
    const layoutUpdBtn = document.getElementById('btnDesignerPageUpdate');
    if (layoutUpdBtn) {
      layoutUpdBtn.classList.remove('layout-update-stale');
      layoutUpdBtn.classList.add('layout-update-fresh');
    }
    if (typeof syncProSimButtonFromDesignerPageState === 'function') syncProSimButtonFromDesignerPageState();
  }
  /** Any Layout tab (settings pane) field commit: Pro Sim + path graph need refresh; Layout Update goes stale (red). */
  function markLayoutPanelFieldDirty() {
    if (typeof markGlobalUpdateStale === 'function') markGlobalUpdateStale();
    if (typeof markDesignerPageUpdateStale === 'function') markDesignerPageUpdateStale();
  }
  /** English multi-line (≤5 aircraft) or summary (6+) for arrival RET failure dock banner. */
  function formatArrRetFailedBannerEnglish(regs) {
    const n = (regs && regs.length) || 0;
    if (n < 1) return '';
    if (n <= 5) {
      return regs.map(function(reg) {
        return String(reg) + ': Runway exit assignment failed.';
      }).join('\n');
    }
    const head = regs.slice(0, 3).join(', ');
    return head + ', etc. — ' + n + ' aircraft failed.';
  }
  /**
   * Pro Sim: allowed only when Update is fresh (green) and no arrival Runway exit (RET) failures
   * (`arrRetFailed` on non-departure legs). RET failure banner is in `#gridLeftFloatingStack`, separate from `#object-info-dock`.
   */
  function getArrRetFailedRegsForProSimUi() {
    const failedRegs = [];
    (state.flights || []).forEach(function(f) {
      if (!f || f.arrDep === 'Dep') return;
      if (f.arrRetFailed) {
        const r = String(f.reg != null && String(f.reg).trim() !== '' ? f.reg : (f.flightNumber || f.id || '')).trim() || '—';
        if (failedRegs.indexOf(r) < 0) failedRegs.push(r);
      }
    });
    return failedRegs;
  }
  function flightApronIntervalsForProSimBlock(f) {
    if (!f || flightBlockedLikeNoWay(f)) return [];
    const segs = typeof normalizeFlightApronStaySegments === 'function' ? normalizeFlightApronStaySegments(f) : [];
    const out = [];
    if (segs.length) {
      const count = segs.length;
      for (let i = 0; i < segs.length; i++) {
        const seg = segs[i];
        const standId = seg && seg.standId != null ? String(seg.standId) : '';
        const t0 = Number(seg && seg.sibtMin);
        const t1 = Number(seg && seg.sobtMin);
        if (!standId || !isFinite(t0) || !isFinite(t1) || t1 <= t0) continue;
        out.push({ f: f, standId: standId, t0: t0, t1: t1, segmentIdx: i, segmentCount: count });
      }
      return out;
    }
    const standId = f.standId != null ? String(f.standId) : '';
    const t0 = f.sibtMin != null ? Number(f.sibtMin) : Number(f.timeMin || 0);
    const t1 = f.sobtMin != null ? Number(f.sobtMin) : (t0 + Number(f.dwellMin || 0));
    return standId && isFinite(t0) && isFinite(t1) && t1 > t0
      ? [{ f: f, standId: standId, t0: t0, t1: t1, segmentIdx: 0, segmentCount: 1 }]
      : [];
  }
  function getApronDuplicatedRegsForProSimUi() {
    const issues = [];
    const seen = new Set();
    const intervalsByStand = {};
    function addIssue(f, reason) {
      if (!f) return;
      const reg = String(f.reg != null && String(f.reg).trim() !== '' ? f.reg : (f.flightNumber || f.id || '')).trim() || '—';
      const key = reg + '|' + reason;
      if (seen.has(key)) return;
      seen.add(key);
      issues.push({ reg: reg, reason: reason });
    }
    (state.flights || []).forEach(function(f) {
      const intervals = flightApronIntervalsForProSimBlock(f);
      intervals.forEach(function(it) {
        const stand = typeof findStandById === 'function' ? findStandById(it.standId) : null;
        if (stand && typeof flightCanUseStandForSegment === 'function' && !flightCanUseStandForSegment(f, stand, it.segmentIdx, it.segmentCount)) {
          if (typeof flightStandAircraftConstraintOk === 'function' && !flightStandAircraftConstraintOk(f, stand)) {
            const apronNo = String((stand.name && String(stand.name).trim()) || stand.id || it.standId || '—').trim();
            addIssue(f, '__apron_size__:' + apronNo);
          } else {
            addIssue(f, 'Invalid apron/building assignment.');
          }
        }
        if (!intervalsByStand[it.standId]) intervalsByStand[it.standId] = [];
        intervalsByStand[it.standId].push(it);
      });
    });
    Object.keys(intervalsByStand).forEach(function(standId) {
      const arr = intervalsByStand[standId];
      for (let i = 0; i < arr.length; i++) {
        for (let j = i + 1; j < arr.length; j++) {
          const a = arr[i], b = arr[j];
          if (a.f && b.f && a.f.id === b.f.id) continue;
          if (a.t0 < b.t1 && b.t0 < a.t1) {
            addIssue(a.f, 'Duplicated apron time window.');
            addIssue(b.f, 'Duplicated apron time window.');
          }
        }
      }
    });
    return issues;
  }
  function getApronStandWindowOverlapRegsForProSimUi() {
    const issues = [];
    const seen = new Set();
    (state.flights || []).forEach(function(f) {
      if (!f || (typeof flightBlockedLikeNoWay === 'function' && flightBlockedLikeNoWay(f))) return;
      const segs = typeof normalizeFlightApronStaySegments === 'function' ? normalizeFlightApronStaySegments(f) : [];
      if (segs.length) {
        for (let j = 0; j < segs.length; j++) {
          const sid = segs[j] && segs[j].standId;
          if (!sid) continue;
          if (typeof flightWouldOverlapStandAssignment === 'function' && flightWouldOverlapStandAssignment(f, sid, j)) {
            const reg = String(f.reg != null && String(f.reg).trim() !== '' ? f.reg : (f.flightNumber || f.id || '')).trim() || '—';
            if (!seen.has(reg)) {
              seen.add(reg);
              issues.push({ reg: reg });
            }
          }
        }
      } else if (f.standId) {
        if (typeof flightWouldOverlapStandAssignment === 'function' && flightWouldOverlapStandAssignment(f, f.standId, null)) {
          const reg = String(f.reg != null && String(f.reg).trim() !== '' ? f.reg : (f.flightNumber || f.id || '')).trim() || '—';
          if (!seen.has(reg)) {
            seen.add(reg);
            issues.push({ reg: reg });
          }
        }
      }
    });
    return issues;
  }
  function flightRegForStandOverlapBanner(f) {
    if (!f) return '—';
    return String(f.reg != null && String(f.reg).trim() !== '' ? f.reg : (f.flightNumber || f.id || '')).trim() || '—';
  }
  function flightStandStayWindowsForOverlapPair(f) {
    if (!f || (typeof flightBlockedLikeNoWay === 'function' && flightBlockedLikeNoWay(f))) return [];
    const out = [];
    const segs = typeof normalizeFlightApronStaySegments === 'function' ? normalizeFlightApronStaySegments(f) : [];
    if (segs.length) {
      for (let j = 0; j < segs.length; j++) {
        const sid = segs[j] && segs[j].standId;
        if (!sid) continue;
        const t0 = Number(segs[j].sibtMin), t1 = Number(segs[j].sobtMin);
        if (!isFinite(t0) || !isFinite(t1) || t1 <= t0) continue;
        out.push({ standId: String(sid), t0: t0, t1: t1 });
      }
      return out;
    }
    if (f.standId) {
      const w = typeof flightScheduleStandWindowMinutes === 'function' ? flightScheduleStandWindowMinutes(f) : null;
      if (w && isFinite(w.sibt) && isFinite(w.sobt) && w.sobt > w.sibt) {
        out.push({ standId: String(f.standId), t0: w.sibt, t1: w.sobt });
      }
    }
    return out;
  }
  function flightPairOverlapsStandWindow(f, g) {
    if (!f || !g || f === g) return false;
    if (typeof flightBlockedLikeNoWay === 'function' && (flightBlockedLikeNoWay(f) || flightBlockedLikeNoWay(g))) return false;
    const fa = flightStandStayWindowsForOverlapPair(f);
    const ga = flightStandStayWindowsForOverlapPair(g);
    for (let i = 0; i < fa.length; i++) {
      const a = fa[i];
      const blockedA = new Set([a.standId].concat(typeof duplicateApronStandIdsForStand === 'function' ? duplicateApronStandIdsForStand(a.standId) : []));
      for (let k = 0; k < ga.length; k++) {
        const b = ga[k];
        if (!blockedA.has(String(b.standId))) continue;
        if (a.t0 < b.t1 && b.t0 < a.t1) return true;
      }
    }
    for (let k = 0; k < ga.length; k++) {
      const b = ga[k];
      const blockedB = new Set([b.standId].concat(typeof duplicateApronStandIdsForStand === 'function' ? duplicateApronStandIdsForStand(b.standId) : []));
      for (let i = 0; i < fa.length; i++) {
        const a = fa[i];
        if (!blockedB.has(String(a.standId))) continue;
        if (a.t0 < b.t1 && b.t0 < a.t1) return true;
      }
    }
    return false;
  }
  function formatStandWindowOverlapBannerDetail() {
    const flights = (state.flights || []).filter(function(f) {
      return f && !(typeof flightBlockedLikeNoWay === 'function' && flightBlockedLikeNoWay(f));
    });
    if (flights.length < 2) return '';
    const parent = flights.map(function(_, i) { return i; });
    function find(i) {
      return parent[i] === i ? i : (parent[i] = find(parent[i]));
    }
    function union(i, j) {
      const pi = find(i), pj = find(j);
      if (pi !== pj) parent[pj] = pi;
    }
    for (let i = 0; i < flights.length; i++) {
      for (let j = i + 1; j < flights.length; j++) {
        if (flightPairOverlapsStandWindow(flights[i], flights[j])) union(i, j);
      }
    }
    const compRegs = new Map();
    for (let i = 0; i < flights.length; i++) {
      const r = find(i);
      const reg = flightRegForStandOverlapBanner(flights[i]);
      if (!compRegs.has(r)) compRegs.set(r, []);
      compRegs.get(r).push(reg);
    }
    const lines = [];
    compRegs.forEach(function(regs) {
      const uniq = [];
      const seen = new Set();
      for (let u = 0; u < regs.length; u++) {
        const rg = regs[u];
        if (!seen.has(rg)) {
          seen.add(rg);
          uniq.push(rg);
        }
