      designerPersist: {
        v: 1,
        globalUpdateFresh: !!state.globalUpdateFresh,
        designerPageUpdateFresh: !!state.designerPageUpdateFresh,
        hasSimulationPlayback: !!state.hasSimulationResult,
        simPlaybackEndCapSec: (state.simPlaybackEndCapSec != null && isFinite(Number(state.simPlaybackEndCapSec)))
          ? Number(state.simPlaybackEndCapSec)
          : null,
        simWindowStartSec: isFinite(Number(state.simWindowStartSec)) ? Number(state.simWindowStartSec) : null,
        simWindowEndSec: isFinite(Number(state.simWindowEndSec)) ? Number(state.simWindowEndSec) : null,
        mapTypeMode: (state.mapTypeMode === 'heatmap') ? 'heatmap' : 'normal',
        heatmapTrafficPhases: Object.assign({}, state.heatmapTrafficPhases || {}),
      },
      simPlaybackPositionsByFlightId: state.hasSimulationResult && state.simPlaybackPositionsByFlightId
        ? state.simPlaybackPositionsByFlightId
        : null,
      simDeadlockGhostPlayback: (function() {
        const dlp = state.simDeadlockGhostPlayback;
        if (!dlp || !Array.isArray(dlp.events) || !dlp.events.length) return null;
        return {
          events: dlp.events.map(function(ev) {
            const o = { t_abs: Number(ev.t_abs), labels: Array.isArray(ev.labels) ? ev.labels.slice() : [] };
            if (ev.focusWorldX != null && isFinite(Number(ev.focusWorldX))) o.focusWorldX = Number(ev.focusWorldX);
            if (ev.focusWorldY != null && isFinite(Number(ev.focusWorldY))) o.focusWorldY = Number(ev.focusWorldY);
            return o;
          }),
          bodyLines: dlp.bodyLines || '',
          resolveCount: isFinite(Number(dlp.resolveCount)) ? Math.floor(Number(dlp.resolveCount)) : 0,
        };
      })(),
      simPathGraph: buildSimPathGraphExport()
    };
  }
  function buildLayout3DViewerPayload() {
    let tSec = Number(state.simTimeSec);
    if (!isFinite(tSec)) tSec = 0;
    const layout = serializeCurrentLayout();
    const flightDrawPoses = [];
    (state.flights || []).forEach(function(f) {
      if (!f) return;
      let pose = null;
      if (typeof getFlightPoseAtTimeForDraw === 'function') {
        pose = getFlightPoseAtTimeForDraw(f, tSec);
      }
      flightDrawPoses.push({
        id: f.id,
        reg: f.reg,
        aircraftType: f.aircraftType,
        code: f.code,
        arrDep: f.arrDep,
        pose: pose && isFinite(pose.x) && isFinite(pose.y) ? { x: pose.x, y: pose.y, dx: pose.dx, dy: pose.dy } : null
      });
    });
    const enrichedFootprints = {
      remote: (state.remoteStands || []).map(function(st) {
        return {
          id: st && st.id,
          name: st && st.name,
          corners: typeof getRemoteStandCorners === 'function' ? getRemoteStandCorners(st) : null
        };
      }).filter(function(r) { return r.corners && r.corners.length >= 3; }),
      pbb: (state.pbbStands || []).map(function(pbb) {
        return {
          id: pbb && pbb.id,
          name: pbb && pbb.name,
          corners: typeof getPBBStandCorners === 'function' ? getPBBStandCorners(pbb) : null
        };
      }).filter(function(r) { return r.corners && r.corners.length >= 3; })
    };
    const enrichedApronLinkPolylines = (state.apronLinks || []).map(function(lk) {
      if (!lk || typeof getApronLinkPolylineWorldPts !== 'function') return null;
      const pts = getApronLinkPolylineWorldPts(lk);
      if (!pts || pts.length < 2) return null;
      return {
        id: lk.id,
        points: pts.map(function(p) { return { x: p[0], y: p[1] }; })
      };
    }).filter(Boolean);
    const payload = {
      version: 1,
      kind: 'grid3dViewer',
      layoutApiUrl: (typeof LAYOUT_API_URL === 'string' && LAYOUT_API_URL) ? LAYOUT_API_URL : '',
      grid3dAssetApiUrl: (typeof GRID3D_ASSET_API_URL === 'string' && GRID3D_ASSET_API_URL) ? GRID3D_ASSET_API_URL : '',
      exportedAt: new Date().toISOString(),
      simTimeSec: tSec,
      viewerConfig: {
        gridMajorInterval: GRID_MAJOR_INTERVAL,
        gridViewBg: GRID_VIEW_BG
      },
      layout: layout,
      flightDrawPoses: flightDrawPoses,
      enrichedFootprints: enrichedFootprints,
      enrichedApronLinkPolylines: enrichedApronLinkPolylines
    };
    try {
      let tiled = null;
      if (typeof exportLayoutGroundTilesFor3D === 'function') tiled = exportLayoutGroundTilesFor3D();
      if (tiled && tiled.tiles && tiled.tiles.length === 4) {
        payload.layoutGroundTiles = tiled;
      } else if (typeof exportLayoutGroundTextureFor3D === 'function') {
        const gt = exportLayoutGroundTextureFor3D();
        if (gt && gt.dataUrl) payload.layoutGroundTexture = gt;
      }
    } catch (eTex) {
      console.warn('exportLayoutGroundTilesFor3D / exportLayoutGroundTextureFor3D failed', eTex);
    }
    return payload;
  }
  function openGrid3DViewerWindow() {
    const tpl = typeof window.__GRID3D_VIEWER_HTML_TEMPLATE__ === 'string' ? window.__GRID3D_VIEWER_HTML_TEMPLATE__ : '';
    if (!tpl || tpl.length < 80) {
      console.error('Grid 3D viewer template missing');
      alert('3D viewer template is not loaded. Ensure pages/Layout_Design/3D/grid3d-viewer.html exists and reload the Layout Design page.');
      return;
    }
    const bootHtml = '<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/><title>Layout 3D</title>' +
      '<style>html,body{margin:0;height:100%;background:#0d0d0f;color:#e2e8f0;font-family:system-ui,sans-serif;overflow:hidden}' +
      '.wrap{display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:100vh;gap:18px;padding:24px;box-sizing:border-box}' +
      '.sp{width:44px;height:44px;border:3px solid rgba(148,163,184,.25);border-top-color:#7c6af7;border-radius:50%;animation:g .85s linear infinite}' +
      '@keyframes g{to{transform:rotate(360deg)}}' +
      '.bar{width:min(360px,86vw);height:4px;border-radius:2px;background:rgba(148,163,184,.2);overflow:hidden}' +
      '.bar>i{display:block;height:100%;width:38%;background:linear-gradient(90deg,#5b52d6,#7c6af7);border-radius:2px;animation:p 1.15s ease-in-out infinite}' +
      '@keyframes p{0%,100%{transform:translateX(-40%)}50%{transform:translateX(200%)}}' +
      '.t{font-size:15px;font-weight:600;color:#f1f5f9;text-align:center}.s{font-size:13px;color:#94a3b8;text-align:center;max-width:360px;line-height:1.45}' +
      '</style></head><body><div class="wrap"><div class="sp"></div><div class="bar"><i></i></div><p class="t">3D 뷰 준비 중</p>' +
      '<p class="s">레이아웃 스냅샷을 만들고 있습니다. 잠시만 기다려 주세요.</p></div></body></html>';
    const g3Base = (typeof GRID3D_ASSET_API_URL === 'string' && GRID3D_ASSET_API_URL.trim()) ? GRID3D_ASSET_API_URL.trim() : '';
    const viewerShellUrl = /^https?:\/\//i.test(g3Base) ? g3Base.replace(/\/$/, '') + '/api/grid3d-viewer-app' : '';
    let w = null;
    let openedViaReceiverShell = false;
    if (viewerShellUrl) {
      try {
        w = window.open(viewerShellUrl, '_blank', 'width=1280,height=840');
        openedViaReceiverShell = !!w;
      } catch (eHttp) {
        console.warn('Grid 3D receiver shell open failed', eHttp);
        w = null;
        openedViaReceiverShell = false;
      }
    }
    if (!w) {
      try {
        w = window.open('data:text/html;charset=utf-8,' + encodeURIComponent(bootHtml), '_blank', 'width=1280,height=840');
      } catch (eData) {
        console.warn('Grid 3D popup data URL failed, using about:blank', eData);
      }
    }
    if (!w) {
      w = window.open('about:blank', '_blank', 'width=1280,height=840');
    }
    if (!w) {
      alert('Popup was blocked. Allow popups for this site to open the 3D viewer.');
      return;
    }
    if (!openedViaReceiverShell) {
      var bootHref = '';
      try {
        bootHref = w.location && w.location.href ? String(w.location.href) : '';
      } catch (eLoc) {
        bootHref = '';
      }
      if (bootHref.indexOf('data:') !== 0) {
        try {
          w.document.open();
          w.document.write(bootHtml);
          w.document.close();
        } catch (eOpen) {
          console.error(eOpen);
          try {
            w.close();
          } catch (eClose) { /* ignore */ }
          alert('Could not open the 3D viewer window.');
          return;
        }
      }
    }
    let payload;
    try {
      payload = typeof buildLayout3DViewerPayload === 'function' ? buildLayout3DViewerPayload() : null;
    } catch (e) {
      console.error('buildLayout3DViewerPayload failed:', e);
      try {
        w.close();
      } catch (eClose2) { /* ignore */ }
      alert('Could not serialize layout for 3D: ' + (e && e.message ? e.message : e));
      return;
    }
    if (!payload || !payload.layout) {
      try {
        w.close();
      } catch (eClose3) { /* ignore */ }
      alert('Could not serialize layout for 3D.');
      return;
    }
    function sendGrid3dInit() {
      try {
        w.postMessage({ kind: 'grid3dViewerInit', payload: payload }, '*');
      } catch (e4) {
        console.error('postMessage to 3D viewer failed:', e4);
        alert('Could not send layout data to the 3D window. Try again or check the browser console.');
      }
    }
    state.grid3dPopupRef = w;
    if (state.prosimBusy) {
      try { w.postMessage({ type: 'prosim:pause' }, '*'); } catch (eP) { /* ignore */ }
    }
    if (!openedViaReceiverShell) {
      try {
        w.document.open();
        w.document.write(tpl);
        w.document.close();
      } catch (e3) {
        console.error(e3);
        try {
          w.close();
        } catch (eClose4) { /* ignore */ }
        alert('Could not write the 3D viewer document.');
        return;
      }
      setTimeout(sendGrid3dInit, 0);
    } else {
      function onShellReady() {
        setTimeout(sendGrid3dInit, 0);
      }
      try {
        if (w.document && w.document.readyState === 'complete') {
          onShellReady();
        } else {
          w.addEventListener('load', function grid3dShellLoad() {
            w.removeEventListener('load', grid3dShellLoad);
            onShellReady();
          });
        }
      } catch (eReady) {
        setTimeout(sendGrid3dInit, 150);
      }
    }
  }
  function getExistingStandBounds() {
    const list = [];
    state.remoteStands.forEach(st => {
      const corners = getRemoteStandCorners(st);
      let left = corners[0][0], right = corners[0][0], top = corners[0][1], bottom = corners[0][1];
      for (let k = 1; k < 4; k++) {
        left = Math.min(left, corners[k][0]); right = Math.max(right, corners[k][0]);
        top = Math.min(top, corners[k][1]); bottom = Math.max(bottom, corners[k][1]);
      }
      list.push({ left, right, top, bottom });
    });
    state.pbbStands.forEach(pbb => {
      const corners = getPBBStandCorners(pbb);
      let left = corners[0][0], right = corners[0][0], top = corners[0][1], bottom = corners[0][1];
      for (let k = 1; k < 4; k++) {
        left = Math.min(left, corners[k][0]); right = Math.max(right, corners[k][0]);
        top = Math.min(top, corners[k][1]); bottom = Math.max(bottom, corners[k][1]);
      }
      list.push({ left, right, top, bottom });
    });
    return list;
  }
  function standOverlapsExisting(bounds) {
    const existing = getExistingStandBounds();
    for (let i = 0; i < existing.length; i++) if (rectsOverlap(bounds, existing[i])) return true;
    return false;
  }
  function dist2(a, b) { const dx = a[0]-b[0], dy = a[1]-b[1]; return dx*dx+dy*dy; }
  function _normalizeTimeToSeconds(value, unit, roundingMode) {
    const raw = Number(value || 0);
    const scaled = unit === 'minutes' ? raw * 60 : raw;
    const rounded = roundingMode === 'round' ? Math.round(scaled) : Math.floor(scaled);
    return Math.max(0, rounded);
  }
  function _splitTotalSeconds(totalSec) {
    const safeSec = Math.max(0, Math.floor(totalSec || 0));
    const h = Math.floor(safeSec / 3600);
    const m = Math.floor((safeSec % 3600) / 60);
    const s = safeSec % 60;
    return {
      h,
      m,
      s,
      hh: (h < 10 ? '0' : '') + h,
      mm: (m < 10 ? '0' : '') + m,
      ss: (s < 10 ? '0' : '') + s,
    };
  }
  function formatMinutesToHHMM(m) {
    const parts = _splitTotalSeconds(_normalizeTimeToSeconds(m, 'minutes', 'floor'));
    return parts.h + ':' + parts.mm;
  }
  function findNearestItem(candidates, getPoint, wx, wy, maxD2) {
    const click = [wx, wy];
    let best = null;
    let bestD2 = maxD2;
    for (let i = 0; i < candidates.length; i++) {
      const c = candidates[i];
      const pt = getPoint(c);
      if (!pt || pt.length < 2) continue;
      const d2 = dist2(pt, click);
      if (d2 < bestD2) {
        bestD2 = d2;
        best = c;
      }
    }
    return best;
  }
  function closestPointOnSegment(p1, p2, p) {
    const [x1,y1]=p1,[x2,y2]=p2,[px,py]=p;
    const dx=x2-x1,dy=y2-y1,len2=dx*dx+dy*dy;
    if (len2===0) return null;
    let t = ((px-x1)*dx+(py-y1)*dy)/len2;
    t = Math.max(0,Math.min(1,t));
    return [x1+t*dx,y1+t*dy];
  }
  function getClosestTerminalEdgePoint(wx, wy) {
    const click = [wx, wy];
    let best = null;
    let bestD2 = Infinity;
    (state.terminals || []).forEach(function(term) {
      if (!term || !term.closed || !Array.isArray(term.vertices) || term.vertices.length < 2) return;
      for (let i = 0; i < term.vertices.length; i++) {
        const v1 = term.vertices[i];
        const v2 = term.vertices[(i + 1) % term.vertices.length];
        const p1 = cellToPixel(v1.col, v1.row);
        const p2 = cellToPixel(v2.col, v2.row);
        const near = closestPointOnSegment(p1, p2, click);
        if (!near) continue;
        const d2 = dist2(near, click);
        if (d2 < bestD2) {
          bestD2 = d2;
          best = { point: near, term: term, edgeIndex: i };
        }
      }
    });
    return best;
  }
  function getPbbBoardingWidthM(pbb) {
    const w = Number(pbb && pbb.boardingWidthM);
    if (isFinite(w) && w > 0) return w;
    return 5;
  }
  function getPbbBoardingHeightM(pbb) {
    const h = Number(pbb && pbb.boardingHeightM);
    if (isFinite(h) && h > 0) return h;
    return 15;
  }
  function getPbbTerminalContactSetbackM(pbb) {
    const v = Number(pbb && pbb.terminalContactSetbackM);
    if (isFinite(v) && v >= 0) return v;
    return 0;
  }
  function getPbbTerminalFrameFromEdge(term, edgeIndex, wallX, wallY) {
    const v1 = term.vertices[edgeIndex], v2 = term.vertices[(edgeIndex + 1) % term.vertices.length];
    const p1 = cellToPixel(v1.col, v1.row), p2 = cellToPixel(v2.col, v2.row);
    const edx = p2[0] - p1[0], edy = p2[1] - p1[1];
    const el = Math.hypot(edx, edy) || 1;
    const tx = edx / el, ty = edy / el;
    let nx = -ty, ny = tx;
    let tcx = 0, tcy = 0;
    term.vertices.forEach(function(v) {
      const q = cellToPixel(v.col, v.row);
      tcx += q[0];
      tcy += q[1];
    });
    tcx /= term.vertices.length;
    tcy /= term.vertices.length;
    const inX = tcx - wallX, inY = tcy - wallY;
    if (nx * inX + ny * inY > 0) {
      nx = -nx;
      ny = -ny;
