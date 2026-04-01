  if (btnRemoteDrawEl) btnRemoteDrawEl.addEventListener('click', function() {
    toggleLayoutDrawMode('remoteDrawing', 'previewRemote', null);
  });
  const btnHoldingPointDrawEl = document.getElementById('btnHoldingPointDraw');
  if (btnHoldingPointDrawEl) btnHoldingPointDrawEl.addEventListener('click', function() {
    toggleLayoutDrawMode('holdingPointDrawing', 'previewHoldingPoint', null);
  });
  const btnApronDrawEl = document.getElementById('btnApronLinkDraw');
  if (btnApronDrawEl) btnApronDrawEl.addEventListener('click', function() {
    toggleLayoutDrawMode('apronLinkDrawing', null, 'apronLinkTemp');
  });

  (function setupRightPanelDragResize() {
    if (!panel || !panelToggle) return;
    const rootStyle = () => getComputedStyle(document.documentElement);
    function readPxVar(name, fallback) {
      const v = parseFloat(rootStyle().getPropertyValue(name));
      return Number.isFinite(v) ? v : fallback;
    }
    function readLenVar(name, fallback) {
      const t = (rootStyle().getPropertyValue(name) || '').trim();
      return t || fallback;
    }
    function parseCssLenToPx(s, vwBase) {
      const str = String(s || '').trim().toLowerCase();
      const n = parseFloat(str);
      if (!Number.isFinite(n)) return vwBase * 0.5;
      if (str.endsWith('vw')) return (n / 100) * vwBase;
      if (str.endsWith('vh')) return (n / 100) * (typeof window !== 'undefined' ? window.innerHeight : 800);
      if (str.endsWith('%')) return (n / 100) * vwBase;
      if (str.endsWith('px')) return n;
      return n;
    }
    function maxPanelPx() {
      const m = readPxVar('--style-right-panel-resize-viewport-margin', 8);
      return Math.max(120, window.innerWidth - m);
    }
    function collapsedPx() { return readPxVar('--style-right-panel-resize-collapsed', 44); }
    function collapseBelowPx() { return readPxVar('--style-right-panel-resize-collapse-below', 96); }
    function minExpandedPx() { return readPxVar('--style-right-panel-resize-min-expanded', 220); }
    let lastExpandedWidthPx = Math.round(parseCssLenToPx(readLenVar('--style-right-panel-width-full', '50vw'), window.innerWidth));
    lastExpandedWidthPx = Math.min(maxPanelPx(), Math.max(minExpandedPx(), lastExpandedWidthPx));
    function syncToolbar(px) {
      document.documentElement.style.setProperty('--layout-toolbar-right', Math.round(px) + 'px');
    }
    function applyCollapsed() {
      panel.classList.add('collapsed');
      panel.style.width = '';
      syncToolbar(collapsedPx());
      panelToggle.textContent = '▶';
    }
    function applyExpandedWidthPx(px) {
      const cap = maxPanelPx();
      let w = Math.min(cap, Math.round(px));
      w = Math.max(minExpandedPx(), w);
      panel.classList.remove('collapsed');
      panel.style.width = w + 'px';
      lastExpandedWidthPx = w;
      syncToolbar(w);
      panelToggle.textContent = '◀';
    }
    function applyDragWidthPx(rawPx) {
      const cap = maxPanelPx();
      const c0 = collapsedPx();
      const below = collapseBelowPx();
      let w = Math.min(cap, Math.max(c0, Math.round(rawPx)));
      if (w < below) {
        panel.classList.add('collapsed');
        panel.style.width = '';
        syncToolbar(c0);
        panelToggle.textContent = '▶';
        return;
      }
      panel.classList.remove('collapsed');
      panel.style.width = w + 'px';
      syncToolbar(w);
      panelToggle.textContent = '◀';
    }
    function finishDragWidthPx(rawPx) {
      const below = collapseBelowPx();
      const cap = maxPanelPx();
      let w = Math.min(cap, Math.max(collapsedPx(), Math.round(rawPx)));
      if (w < below) {
        applyCollapsed();
        return;
      }
      w = Math.min(cap, Math.max(minExpandedPx(), w));
      applyExpandedWidthPx(w);
    }
    applyExpandedWidthPx(lastExpandedWidthPx);
    let dragStartClientX = 0;
    let dragStartWidth = 0;
    let lastMoveClientX = 0;
    let dragMoved = false;
    let resizePointerActive = false;
    let suppressToggleClick = false;
    const CLICK_MAX_MOVE = _interactionConfigNum('clickMaxMovePx', 6);
    function onResizeWindow() {
      if (panel.classList.contains('collapsed')) {
        syncToolbar(collapsedPx());
        return;
      }
      const rw = panel.getBoundingClientRect().width;
      const cap = maxPanelPx();
      if (rw > cap) applyExpandedWidthPx(cap);
      else syncToolbar(rw);
    }
    window.addEventListener('resize', onResizeWindow);
    panelToggle.addEventListener('click', function(ev) {
      if (suppressToggleClick) {
        ev.preventDefault();
        ev.stopImmediatePropagation();
        suppressToggleClick = false;
      }
    }, true);
    panelToggle.addEventListener('pointerdown', function(ev) {
      if (ev.pointerType === 'mouse' && ev.button !== 0) return;
      ev.preventDefault();
      dragMoved = false;
      resizePointerActive = true;
      dragStartClientX = ev.clientX;
      lastMoveClientX = ev.clientX;
      const c0 = collapsedPx();
      dragStartWidth = panel.classList.contains('collapsed') ? c0 : panel.getBoundingClientRect().width;
      panel.classList.add('panel-resize-dragging');
      try { panelToggle.setPointerCapture(ev.pointerId); } catch (e) {}
    });
    panelToggle.addEventListener('pointermove', function(ev) {
      if (!resizePointerActive) return;
      if (Math.abs(ev.clientX - dragStartClientX) > CLICK_MAX_MOVE) dragMoved = true;
      lastMoveClientX = ev.clientX;
      const w = dragStartWidth + (dragStartClientX - ev.clientX);
      applyDragWidthPx(w);
    });
    function endPointerDrag(ev) {
      if (!resizePointerActive) return;
      resizePointerActive = false;
      panel.classList.remove('panel-resize-dragging');
      try { if (ev && ev.pointerId != null) panelToggle.releasePointerCapture(ev.pointerId); } catch (e) {}
      if (!dragMoved) {
        if (panel.classList.contains('collapsed')) {
          applyExpandedWidthPx(lastExpandedWidthPx);
        } else {
          lastExpandedWidthPx = Math.max(minExpandedPx(), Math.min(maxPanelPx(), panel.getBoundingClientRect().width));
          applyCollapsed();
        }
        dragMoved = false;
        return;
      }
      suppressToggleClick = true;
      const endX = ev && Number.isFinite(ev.clientX) ? ev.clientX : lastMoveClientX;
      const w = dragStartWidth + (dragStartClientX - endX);
      finishDragWidthPx(w);
      dragMoved = false;
    }
    panelToggle.addEventListener('pointerup', endPointerDrag);
    panelToggle.addEventListener('pointercancel', endPointerDrag);
    panelToggle.addEventListener('lostpointercapture', function(ev) {
      if (resizePointerActive) endPointerDrag(ev);
    });
  })();

  function renderObjectList() {
    if (!objectListEl) return;
    const mode = settingModeSelect.value;
    const seen = {};
    function uniqueTitle(baseName) {
      return baseName;
    }
    const items = [];
    if (mode === 'terminal') {
      state.terminals.forEach((t, idx) => {
        if (seen['terminal_' + t.id]) return;
        seen['terminal_' + t.id] = true;
        const areaM2 = t.vertices && t.vertices.length >= 3 ? polygonAreaM2(t.vertices) : 0;
        const floors = t.floors != null ? Math.max(1, parseInt(t.floors, 10) || 1) : 1;
        const f2fRaw = t.floorToFloor != null ? Number(t.floorToFloor) : (t.floorHeight != null ? Number(t.floorHeight) : 4);
        const f2f = Math.max(0.5, f2fRaw || 4);
        const floorH = t.floorHeight != null ? Number(t.floorHeight) || (floors * f2f) : (floors * f2f);
        const dep = t.departureCapacity != null ? t.departureCapacity : 0;
        const arr = t.arrivalCapacity != null ? t.arrivalCapacity : 0;
        const baseName = (t.name && t.name.trim()) ? t.name.trim() : ('Building ' + (idx + 1));
        const buildingTheme = getBuildingTheme(t);
        items.push({
          type: 'terminal',
          id: t.id,
          title: uniqueTitle('Building | ' + baseName),
          tag: 'Height ' + floorH.toFixed(1) + ' m',
          details:
            'Type: ' + buildingTheme.label +
            '<br>' +
            'Area: ' + areaM2.toFixed(1) + ' m²' +
            '<br>Height: ' + floorH.toFixed(1) + ' m' +
            '<br>Floors: ' + floors +
            '<br>Total floor area: ' + (areaM2 * floors).toFixed(1) + ' m²' +
            '<br>Departure: ' + dep +
            '<br>Arrival: ' + arr
        });
      });
    } else if (mode === 'pbb') {
      state.pbbStands.forEach((pbb, idx) => {
        if (seen['pbb_' + pbb.id]) return;
        seen['pbb_' + pbb.id] = true;
        const baseName = (pbb.name && pbb.name.trim()) ? pbb.name.trim() : ('Contact Stand ' + (idx + 1));
        items.push({
          type: 'pbb',
          id: pbb.id,
          title: uniqueTitle('Contact Stand | ' + baseName),
          tag: 'Category ' + (pbb.category || 'C'),
          details: 'Edge cell: (' + pbb.edgeCol + ',' + pbb.edgeRow + ')'
        });
      });
    } else if (mode === 'remote') {
      state.remoteStands.forEach((st, idx) => {
        if (seen['remote_' + st.id]) return;
        seen['remote_' + st.id] = true;
        const baseName = (st.name && st.name.trim()) ? st.name.trim() : ('R' + String(idx + 1).padStart(3, '0'));
        let allowedLabel = 'All (by proximity)';
        if (Array.isArray(st.allowedTerminals) && st.allowedTerminals.length) {
          const terms = makeUniqueNamedCopy(state.terminals || [], 'name').map(function(t) { return {
            id: t.id,
            name: (t.name || '').trim() || 'Building'
          }; });
          const names = st.allowedTerminals.map(function(id) {
            const tt = terms.find(function(t) { return t.id === id; });
            return tt ? tt.name : id;
          });
          if (names.length) allowedLabel = names.join(', ');
        }
        const [rcx, rcy] = getRemoteStandCenterPx(st);
        const rcol = rcx / CELL_SIZE;
        const rrow = rcy / CELL_SIZE;
        items.push({
          type: 'remote',
          id: st.id,
          title: uniqueTitle('Remote stand | ' + baseName),
          tag: 'Category ' + (st.category || 'C'),
          details:
            'Category: ' + (st.category || '—') +
            '<br>Position: (' + rcol.toFixed(1) + ',' + rrow.toFixed(1) + ')' +
            '<br>Angle: ' + normalizeAngleDeg(st.angleDeg != null ? st.angleDeg : 0).toFixed(0) + '°' +
            '<br>available buildings: ' + allowedLabel
        });
      });
    } else if (isPathLayoutMode(mode)) {
      const wantPt = pathTypeFromLayoutMode(mode);
      state.taxiways.forEach((tw, idx) => {
        if (seen['taxiway_' + tw.id]) return;
        const pt = tw.pathType || 'taxiway';
        if (pt !== wantPt) return;
        seen['taxiway_' + tw.id] = true;
        const baseName = (tw.name && tw.name.trim()) ? tw.name.trim() : ('Taxiway ' + (idx + 1));
        const dirVal = getTaxiwayDirection(tw);
        const dirLabel = dirVal === 'clockwise' ? 'CW' : (dirVal === 'counter_clockwise' ? 'CCW' : 'Both');
        let lengthM = 0;
        if (tw.vertices && tw.vertices.length >= 2) {
          for (let i = 1; i < tw.vertices.length; i++) {
            const v0 = tw.vertices[i - 1];
            const v1 = tw.vertices[i];
            const dx = v1.col - v0.col;
            const dy = v1.row - v0.row;
            lengthM += CELL_SIZE * Math.hypot(dx, dy);
          }


        }
        const widthDefault = tw.pathType === 'runway'
          ? RUNWAY_PATH_DEFAULT_WIDTH
          : (tw.pathType === 'runway_exit' ? RUNWAY_EXIT_DEFAULT_WIDTH : TAXIWAY_DEFAULT_WIDTH);
        const widthVal = tw.width != null ? tw.width : widthDefault;
        const serTw = serializeTaxiwayWithEndpoints(tw);
        const startStr = serTw.start_point != null ? '(' + serTw.start_point.col + ',' + serTw.start_point.row + ')' : '—';
        const endStr = serTw.end_point != null ? '(' + serTw.end_point.col + ',' + serTw.end_point.row + ')' : '—';
        const heading = tw.pathType === 'runway' ? 'Runway' : (tw.pathType === 'runway_exit' ? 'Runway Taxiway' : 'Taxiway');
        const avgVel = (typeof tw.avgMoveVelocity === 'number' && isFinite(tw.avgMoveVelocity) && tw.avgMoveVelocity > 0) ? tw.avgMoveVelocity : 10;
        const maxExit = (tw.pathType === 'runway_exit' && typeof tw.maxExitVelocity === 'number' && isFinite(tw.maxExitVelocity) && tw.maxExitVelocity > 0) ? tw.maxExitVelocity : null;
        const minExit = (tw.pathType === 'runway_exit' && typeof tw.minExitVelocity === 'number' && isFinite(tw.minExitVelocity) && tw.minExitVelocity > 0)
          ? (maxExit != null && tw.minExitVelocity > maxExit ? maxExit : tw.minExitVelocity)
          : null;
        const minArrDisplay = tw.pathType === 'runway'
          ? ((typeof tw.minArrVelocity === 'number' && isFinite(tw.minArrVelocity) && tw.minArrVelocity > 0)
            ? Math.max(1, Math.min(150, tw.minArrVelocity))
            : 15)
          : null;
        items.push({
          type: 'taxiway',
          id: tw.id,
          title: uniqueTitle(heading + ' | ' + baseName),
          tag: dirLabel,
          details:
            'Length: ' + lengthM.toFixed(0) + ' m' +
            '<br>Points: ' + tw.vertices.length +
            '<br>Width: ' + widthVal + ' m' +
            (maxExit != null ? '<br>Max exit velocity: ' + maxExit + ' m/s' : '') +
            (minExit != null ? '<br>Min exit velocity: ' + minExit + ' m/s' : '') +
            (minArrDisplay != null ? '<br>Min arr velocity: ' + minArrDisplay + ' m/s' : '') +
            (tw.pathType === 'runway' ? '<br>Line up: ' + getEffectiveRunwayLineupDistM(tw) + ' m (start→end)' : '') +
            (tw.pathType === 'taxiway' ? '<br>Avg move velocity: ' + avgVel + ' m/s' : '') +
            '<br>Start point: ' + startStr +
            '<br>End point: ' + endStr
        });
      });
    } else if (mode === 'holdingPoint') {
      (state.holdingPoints || []).forEach(function(hp, idx) {
        if (!hp || seen['hp_' + hp.id]) return;
        seen['hp_' + hp.id] = true;
        const kindLabel = holdingPointKindDisplayLabel(hp.hpKind);
        const baseName = (hp.name && hp.name.trim()) ? hp.name.trim() : (kindLabel + ' ' + (idx + 1));
        const cx = Number(hp.x), cy = Number(hp.y);
        const col = cx / CELL_SIZE, row = cy / CELL_SIZE;
        const tagShort = normalizeHoldingPointKind(hp.hpKind) === 'runway_holding' ? 'RHP' : 'IHP';
        items.push({
          type: 'holdingPoint',
          id: hp.id,
          title: uniqueTitle(kindLabel + ' | ' + baseName),
          tag: tagShort + ' · ' + c2dHoldingPointDiameterM().toFixed(0) + ' m',
          details:
            'Type: ' + kindLabel +
            '<br>Position (cell): (' + col.toFixed(1) + ', ' + row.toFixed(1) + ')' +
            '<br>World: (' + cx.toFixed(0) + ', ' + cy.toFixed(0) + ')'
        });
      });
    } else if (mode === 'apronTaxiway') {
      state.apronLinks.forEach((lk, idx) => {
        if (seen['apron_' + lk.id]) return;
        seen['apron_' + lk.id] = true;
        const stand = findStandById(lk.pbbId);
        const tw = state.taxiways.find(t => t.id === lk.taxiwayId);
        const title = getApronLinkDisplayName(lk);
        const standLabel = stand && stand.name ? stand.name : lk.pbbId;
        const details = 'Stand: ' + standLabel +
          ', Taxiway: ' + (tw && tw.name ? tw.name : lk.taxiwayId);
        items.push({
          type: 'apronLink',
          id: lk.id,
          title: uniqueTitle('Apron–Taxiway | ' + title),
          tag: 'Apron–Taxiway',
          details
        });
      });
    } else if (mode === 'edge') {
      rebuildDerivedGraphEdges();
      (state.derivedGraphEdges || []).forEach(function(ed) {
        items.push({
          type: 'layoutEdge',
          id: ed.id,
          title: 'Edge | ' + getLayoutEdgeDisplayName(ed),
          tag: 'Graph',
          details:
            'Length (graph): ' + Math.round(ed.dist) +
            '<br>Pixel span: (' + ed.x1.toFixed(0) + ', ' + ed.y1.toFixed(0) + ') → (' + ed.x2.toFixed(0) + ', ' + ed.y2.toFixed(0) + ')' +
            '<br>Polyline points: ' + ((ed.pts && ed.pts.length) ? ed.pts.length : 2) +
            '<br>Node indices: ' + ed.fromIdx + ' → ' + ed.toIdx,
          noDelete: true
        });
      });
    }
    if (!items.length) {
      objectListEl.innerHTML = '<div class="obj-item">No objects yet.</div>';
      return;
    }
    objectListEl.innerHTML = items.map(it => (
      '<div class="obj-item" data-type="' + it.type + '" data-id="' + it.id + '">' +
        '<div class="obj-item-header">' +
          '<span class="obj-item-title">' + it.title + '</span>' +
          '<span class="obj-item-tag">' + it.tag + '</span>' +
          '<button type="button" class="obj-item-delete" title="Delete"' + (it.noDelete ? ' style="display:none" tabindex="-1" aria-hidden="true"' : '') + '>×</button>' +
        '</div>' +
        '<div class="obj-item-details">' + it.details + '</div>' +
      '</div>'
    )).join('');
    const listItems = objectListEl.querySelectorAll('.obj-item');
    listItems.forEach(el => {
      const type = el.getAttribute('data-type');
      const id = el.getAttribute('data-id');
      el.querySelector('.obj-item-delete').addEventListener('click', function(ev) {
        ev.stopPropagation();
        pushUndo();
        removeLayoutObjectFromState(type, id);
        if (state.selectedObject && state.selectedObject.type === type && state.selectedObject.id === id)
          state.selectedObject = null;
        if (type === 'terminal' && state.currentTerminalId === id) {
          state.currentTerminalId = state.terminals.length ? state.terminals[0].id : null;
          if (state.terminalDrawingId === id) {
            state.terminalDrawingId = null;
            state.layoutPathDrawPointer = null;
          }
        }
        if (type === 'taxiway' && state.taxiwayDrawingId === id) {
          state.taxiwayDrawingId = null;
          state.layoutPathDrawPointer = null;
        }
        syncPanelFromState();
        updateObjectInfo();
        if (typeof redrawLayoutAfterEdit === 'function') redrawLayoutAfterEdit();
        else if (typeof updateAllFlightPaths === 'function') updateAllFlightPaths(); else draw();
      });
      el.addEventListener('click', function(ev) {
        if (ev.target.classList.contains('obj-item-delete')) return;
        const typ = this.getAttribute('data-type');
        const idr = this.getAttribute('data-id');
        if (typ === 'layoutEdge') rebuildDerivedGraphEdges();
        const obj = findLayoutObjectByListType(typ, idr);
        if (!obj) return;
        const wasExpanded = this.classList.contains('expanded');
        listItems.forEach(li => li.classList.remove('selected', 'expanded'));
        if (!wasExpanded) {
          this.classList.add('selected', 'expanded');
          state.flightPathRevealFlightId = null;
          state.selectedObject = { type: typ, id: idr, obj };
          if (typ === 'terminal') state.currentTerminalId = idr;
          syncPanelFromState();
          updateObjectInfo();
        } else {
          objectInfoEl.textContent = 'Select an object on the grid or from the list.';
        }
        draw();
      });
    });
    if (state.selectedObject) {
      const sel = objectListEl.querySelector('.obj-item[data-type="' + state.selectedObject.type + '"][data-id="' + state.selectedObject.id + '"]');
      if (sel) sel.classList.add('selected', 'expanded');
    }
  }

  function updateObjectInfo() {
    if (state.selectedObject) {
      const o = state.selectedObject.obj;
      if (state.selectedObject.type === 'terminal') {
        const areaM2 = o.vertices && o.vertices.length >= 3 ? polygonAreaM2(o.vertices) : 0;
        const floors = o.floors != null ? Math.max(1, parseInt(o.floors, 10) || 1) : 1;
        const f2fRaw = o.floorToFloor != null ? Number(o.floorToFloor) : (o.floorHeight != null ? Number(o.floorHeight) : 4);
        const f2f = Math.max(0.5, f2fRaw || 4);
        const floorH = o.floorHeight != null ? Number(o.floorHeight) || (floors * f2f) : (floors * f2f);
        const totalArea = areaM2 * floors;
        const dep = o.departureCapacity != null ? o.departureCapacity : 0;
        const arr = o.arrivalCapacity != null ? o.arrivalCapacity : 0;
        objectInfoEl.innerHTML = '<strong>Building</strong><br>Name: ' + (o.name || o.id) + '<br>Type: ' + getBuildingTypeLabel(o.buildingType) + '<br>Vertices: ' + (o.vertices ? o.vertices.length : 0) +
          '<br>Footprint area: ' + areaM2.toFixed(1) + ' m²<br>Height: ' + floorH.toFixed(1) + ' m (Floors: ' + floors + ' × ' + f2f.toFixed(1) + ' m)' +
          '<br>Total floor area: ' + totalArea.toFixed(1) + ' m²' +
          '<br>Departure capacity: ' + dep + '<br>Arrival capacity: ' + arr;
      } else if (state.selectedObject.type === 'pbb') {
        objectInfoEl.innerHTML = '<strong>Contact Stand</strong><br>Name: ' + (o.name || '—') + '<br>Constraint: ' + (getStandCategoryMode(o) === 'aircraft' ? 'Aircraft Type' : ('ICAO ' + (o.category || '—'))) + '<br>PBB count: ' + Math.max(1, parseInt(o.pbbCount, 10) || 1) + '<br>Edge cell: (' + o.edgeCol + ',' + o.edgeRow + ')';
      } else if (state.selectedObject.type === 'remote') {
        let allowedLabel = 'All (by proximity)';
        if (Array.isArray(o.allowedTerminals) && o.allowedTerminals.length) {
          const terms = makeUniqueNamedCopy(state.terminals || [], 'name').map(function(t) { return {
            id: t.id,
            name: (t.name || '').trim() || 'Building'
          }; });
          const names = o.allowedTerminals.map(function(id) {
            const tt = terms.find(function(t) { return t.id === id; });
            return tt ? tt.name : id;
          });
          if (names.length) allowedLabel = names.join(', ');
        }
        const remotePx = getRemoteStandCenterPx(o);
        const remoteCell = [remotePx[0] / CELL_SIZE, remotePx[1] / CELL_SIZE];
        objectInfoEl.innerHTML =
          '<strong>Remote stand</strong>' +
          '<br>Name: ' + (o.name || '—') +
          '<br>Constraint: ' + (getStandCategoryMode(o) === 'aircraft' ? 'Aircraft Type' : ('ICAO ' + (o.category || '—'))) +
          '<br>Cell: (' + remoteCell[0].toFixed(1) + ',' + remoteCell[1].toFixed(1) + ')' +
          '<br>available buildings: ' + allowedLabel;
      } else if (state.selectedObject.type === 'holdingPoint') {
        const hx = Number(o.x), hy = Number(o.y);
        const hCol = hx / CELL_SIZE, hRow = hy / CELL_SIZE;
        objectInfoEl.innerHTML =
          '<strong>' + holdingPointKindDisplayLabel(o.hpKind) + '</strong>' +
          '<br>Name: ' + (o.name || '—') +
          '<br>Diameter: ' + c2dHoldingPointDiameterM().toFixed(0) + ' m' +
          '<br>Cell: (' + hCol.toFixed(1) + ', ' + hRow.toFixed(1) + ')' +
          '<br>World: (' + hx.toFixed(0) + ', ' + hy.toFixed(0) + ')';
      }
      else if (state.selectedObject.type === 'taxiway') {
        const dirVal = getTaxiwayDirection(o);
        const dirLabel = dirVal === 'clockwise' ? 'Clockwise' : (dirVal === 'counter_clockwise' ? 'Counter Clockwise' : 'Both');
        const heading = o.pathType === 'runway' ? 'Runway' : (o.pathType === 'runway_exit' ? 'Runway Taxiway' : 'Taxiway');
        const ser = serializeTaxiwayWithEndpoints(o);
        const startStr = ser.start_point != null ? '(' + ser.start_point.col + ', ' + ser.start_point.row + ')' : '—';
        const endStr = ser.end_point != null ? '(' + ser.end_point.col + ', ' + ser.end_point.row + ')' : '—';
        const avgVel = (typeof o.avgMoveVelocity === 'number' && isFinite(o.avgMoveVelocity) && o.avgMoveVelocity > 0) ? o.avgMoveVelocity : 10;
        const minArr = (o.pathType === 'runway')
          ? ((typeof o.minArrVelocity === 'number' && isFinite(o.minArrVelocity) && o.minArrVelocity > 0) ? Math.max(1, Math.min(150, o.minArrVelocity)) : 15)
          : null;
        const lineupStr = (o.pathType === 'runway') ? (String(getEffectiveRunwayLineupDistM(o)) + ' m (from start toward end)') : '';
        const maxEx = (o.pathType === 'runway_exit' && typeof o.maxExitVelocity === 'number' && isFinite(o.maxExitVelocity) && o.maxExitVelocity > 0) ? o.maxExitVelocity : null;
        const minEx = (o.pathType === 'runway_exit' && typeof o.minExitVelocity === 'number' && isFinite(o.minExitVelocity) && o.minExitVelocity > 0) ? o.minExitVelocity : null;
        objectInfoEl.innerHTML = '<strong>' + heading + '</strong><br>Name: ' + (o.name || '—') +
          '<br>Direction: ' + dirLabel +
          '<br>Width: ' + (o.width != null ? o.width : 23) + ' m' +
          (o.pathType === 'taxiway' ? '<br>Avg move velocity: ' + avgVel + ' m/s' : '') +
          (minArr != null ? '<br>Min arr velocity: ' + minArr + ' m/s' : '') +
          (o.pathType === 'runway' ? '<br>Line up: ' + lineupStr : '') +
          (maxEx != null ? '<br>Max exit velocity: ' + maxEx + ' m/s' : '') +
          (minEx != null ? '<br>Min exit velocity: ' + minEx + ' m/s' : '') +
          '<br>Points: ' + (o.vertices ? o.vertices.length : 0) +
          '<br>Start point: ' + startStr + '<br>End point: ' + endStr;
      } else if (state.selectedObject.type === 'apronLink') {
        const lk = o;
        const stand = findStandById(lk.pbbId);
        const tw = state.taxiways.find(function(t) { return t.id === lk.taxiwayId; });
        objectInfoEl.innerHTML =
          '<strong>Apron Taxiway</strong><br>' +
          'Name: ' + getApronLinkDisplayName(lk) +
          '<br>Stand: ' + (stand && stand.name ? stand.name : lk.pbbId) +
          '<br>Taxiway: ' + (tw && tw.name ? tw.name : lk.taxiwayId) +
          '<br>Link point: (' + Number(lk.tx).toFixed(0) + ', ' + Number(lk.ty).toFixed(0) + ')';
      } else if (state.selectedObject.type === 'layoutEdge') {
        const ed = state.selectedObject.obj;
        objectInfoEl.innerHTML =
          '<strong>Edge (derived)</strong><br>' +
          'Name: ' + getLayoutEdgeDisplayName(ed) +
          '<br>Graph length: ' + (ed && ed.dist != null ? Math.round(ed.dist) : '—') +
          '<br>Nodes: ' + (ed ? ed.fromIdx + ' → ' + ed.toIdx : '—') +
          '<br>Span (px): (' + (ed ? ed.x1.toFixed(0) : '—') + ', ' + (ed ? ed.y1.toFixed(0) : '—') + ') → (' + (ed ? ed.x2.toFixed(0) : '—') + ', ' + (ed ? ed.y2.toFixed(0) : '—') + ')' +
          '<br>Polyline points: ' + (ed && ed.pts ? ed.pts.length : 2);
      } else if (state.selectedObject.type === 'flight') {
        const dir = o.arrDep === 'Dep' ? 'Departure' : 'Arrival';
        const sibt = formatMinutesToHHMMSS(o.sibtMin_d != null ? o.sibtMin_d : (o.timeMin != null ? o.timeMin : 0));
        const sobt = formatMinutesToHHMMSS(o.sobtMin_d != null ? o.sobtMin_d : ((o.timeMin != null ? o.timeMin : 0) + (o.dwellMin != null ? o.dwellMin : 0)));
        const ac = typeof getAircraftInfoByType === 'function' ? getAircraftInfoByType(o.aircraftType) : null;
        const acName = ac ? (ac.name || ac.id || '') : (o.aircraftType || '—');
        const codeIcao = (ac && ac.icao) ? ac.icao : (o.code || '—');
        const icaoJhl = (ac && ac.icaoJHL) ? ac.icaoJHL : '—';
        const recatEu = (ac && ac.recatEu) ? ac.recatEu : '—';
        objectInfoEl.innerHTML =
          '<strong>Flight</strong><br>' +
          'Type: ' + dir +
          '<br>SIBT: ' + sibt + ' &nbsp; SOBT: ' + sobt +
          '<br>Aircraft: ' + (acName || '—') +
          '<br>Code(ICAO): ' + (codeIcao || '—') + ' &nbsp; ICAO(J/H/M/L): ' + (icaoJhl || '—') + ' &nbsp; RECAT-EU: ' + (recatEu || '—') +
          '<br>Reg: ' + (o.reg || '—') +
          '<br>Airline Code: ' + (o.airlineCode || '—') + ' &nbsp; Flight Number: ' + (o.flightNumber || '—') +
          '<br>Dwell (Arr only): ' + (o.dwellMin || 0) + ' min';
      }
    } else
      objectInfoEl.textContent = 'Select an object on the grid or from the list.';
    renderObjectList();
  }

  function reset2DView() {
    let w = 0, h = 0;
    const rect = container.getBoundingClientRect();
    w = Number(rect.width) || 0;
    h = Number(rect.height) || 0;
    if (w <= 0 || h <= 0) {
      if (canvas) {
        w = canvas.clientWidth || canvas.width || 800;
        h = canvas.clientHeight || canvas.height || 600;
      } else {
        w = 800;
        h = 600;
      }
    }
    w = Math.max(1, w);
    h = Math.max(1, h);
    const maxX = GRID_COLS * CELL_SIZE;
    const maxY = GRID_ROWS * CELL_SIZE;
    const scaleX = w / maxX;
    const scaleY = h / maxY;
    const s = Math.min(scaleX, scaleY) * 0.9;
    state.scale = s;
    state.panX = (w - maxX * s) / 2;
    state.panY = (h - maxY * s) / 2;
    draw();
  }

  function resizeCanvas() {
    if (!container || !canvas || !ctx) return;
    const rect = container.getBoundingClientRect();
    const w = Math.max(1, Number(rect.width) || 0);
    const h = Math.max(1, Number(rect.height) || 0);
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    invalidateGridUnderlay();
    safeDraw();
  }

  let _gridUnderlayCanvas = null;
  let _gridUnderlayDirty = true;
  function invalidateGridUnderlay() { _gridUnderlayDirty = true; }
  function rebuildGridUnderlay() {
    const maxX = GRID_COLS * CELL_SIZE, maxY = GRID_ROWS * CELL_SIZE;
    if (!_gridUnderlayCanvas) _gridUnderlayCanvas = document.createElement('canvas');
    _gridUnderlayCanvas.width = Math.max(1, Math.floor(maxX * dpr));
    _gridUnderlayCanvas.height = Math.max(1, Math.floor(maxY * dpr));
    const uctx = _gridUnderlayCanvas.getContext('2d');
    uctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    uctx.fillStyle = GRID_VIEW_BG;
    uctx.fillRect(0, 0, maxX, maxY);
    if (state.layoutImageOverlay && layoutImageBitmap) {
      const overlay = state.layoutImageOverlay;
      const [imgX, imgY] = cellToPixel(overlay.topLeftCol, overlay.topLeftRow);
      uctx.save();
      uctx.globalAlpha = state.showImage ? clampLayoutImageOpacity(overlay.opacity) : 0;
      uctx.imageSmoothingEnabled = true;
      uctx.drawImage(
        layoutImageBitmap,
        imgX,
        imgY,
        clampLayoutImageSize(overlay.widthM, GRID_LAYOUT_IMAGE_DEFAULTS.widthM),
        clampLayoutImageSize(overlay.heightM, GRID_LAYOUT_IMAGE_DEFAULTS.heightM)
      );
      uctx.restore();
    }
    _gridUnderlayDirty = false;
  }

  function drawGrid() {
    const w = canvas.width / dpr, h = canvas.height / dpr;
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = GRID_VIEW_BG;
    ctx.fillRect(0, 0, w, h);
    ctx.restore();
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    const maxX = GRID_COLS * CELL_SIZE, maxY = GRID_ROWS * CELL_SIZE;
    if (_gridUnderlayDirty) rebuildGridUnderlay();
    ctx.drawImage(_gridUnderlayCanvas, 0, 0, maxX, maxY);
    if (!state.showGrid) {
      ctx.restore();
      return;
    }
    const drawMinor = !(GRID_MINOR_GRID_MIN_SCALE > 0 && state.scale < GRID_MINOR_GRID_MIN_SCALE);
    const marginWorld = GRID_DRAW_VIEWPORT_MARGIN_CELLS * CELL_SIZE;
    const s = state.scale || 1;
    const minWx = (0 - state.panX) / s - marginWorld;
    const maxWx = (w - state.panX) / s + marginWorld;
    const minWy = (0 - state.panY) / s - marginWorld;
    const maxWy = (h - state.panY) / s + marginWorld;
    const cMin = Math.max(0, Math.floor(minWx / CELL_SIZE));
    const cMax = Math.min(GRID_COLS, Math.ceil(maxWx / CELL_SIZE));
    const rMin = Math.max(0, Math.floor(minWy / CELL_SIZE));
    const rMax = Math.min(GRID_ROWS, Math.ceil(maxWy / CELL_SIZE));
    for (let c = cMin; c <= cMax; c++) {
      const isMajor = (c % GRID_MAJOR_INTERVAL === 0);
      if (!isMajor && !drawMinor) continue;
      const x = c * CELL_SIZE;
      ctx.strokeStyle = isMajor
        ? ('rgba(' + GRID_MAJOR_LINE_RGB + ',' + GRID_MAJOR_LINE_OPACITY + ')')
        : ('rgba(' + GRID_MINOR_LINE_RGB + ',' + GRID_MINOR_LINE_OPACITY + ')');
      ctx.lineWidth = isMajor ? GRID_MAJOR_LINE_WIDTH : GRID_MINOR_LINE_WIDTH;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, maxY);
      ctx.stroke();
    }
    for (let r = rMin; r <= rMax; r++) {
      const isMajor = (r % GRID_MAJOR_INTERVAL === 0);
      if (!isMajor && !drawMinor) continue;
      const y = r * CELL_SIZE;
      ctx.strokeStyle = isMajor
        ? ('rgba(' + GRID_MAJOR_LINE_RGB + ',' + GRID_MAJOR_LINE_OPACITY + ')')
        : ('rgba(' + GRID_MINOR_LINE_RGB + ',' + GRID_MINOR_LINE_OPACITY + ')');
      ctx.lineWidth = isMajor ? GRID_MAJOR_LINE_WIDTH : GRID_MINOR_LINE_WIDTH;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(maxX, y);
      ctx.stroke();
    }
    ctx.fillStyle = '#aaa';
    ctx.font = '10px system-ui';
    ctx.fillText('0,0', 4, 2);
    const cx = (GRID_COLS * CELL_SIZE) / 2;
    const cy = (GRID_ROWS * CELL_SIZE) / 2;
    ctx.beginPath();
    ctx.fillStyle = '#ef4444';
    ctx.arc(cx, cy, CELL_SIZE * 0.15, 0, Math.PI * 2);
    ctx.fill();
    if (state.hoverCell != null) {
      const hc = state.hoverCell;
      const hx = hc.col * CELL_SIZE;
      const hy = hc.row * CELL_SIZE;
      ctx.beginPath();
      ctx.fillStyle = 'rgba(248, 113, 113, 0.45)';
      ctx.arc(hx, hy, CELL_SIZE * 0.2, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  function drawPolygonHatch(points, strokeStyle, spacingPx) {
    if (!Array.isArray(points) || points.length < 3) return;
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    points.forEach(function(p) {
      minX = Math.min(minX, p[0]);
      maxX = Math.max(maxX, p[0]);
      minY = Math.min(minY, p[1]);
      maxY = Math.max(maxY, p[1]);
    });
    const span = Math.max(maxX - minX, maxY - minY);
    const pad = span + Math.max(40, spacingPx * 2);
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < points.length; i++) ctx.lineTo(points[i][0], points[i][1]);
    ctx.closePath();
    ctx.clip();
    ctx.strokeStyle = strokeStyle;
    ctx.lineWidth = 1.2;
    ctx.setLineDash([]);
    for (let offset = minX - pad; offset <= maxX + pad; offset += spacingPx) {
      ctx.beginPath();
      ctx.moveTo(offset, maxY + pad);
      ctx.lineTo(offset + (maxY - minY) + pad, minY - pad);
      ctx.stroke();
    }
    ctx.restore();
  }
  function drawTerminals() {
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    state.terminals.forEach(term => {
      const isDrawingTerm = state.terminalDrawingId === term.id;
      if (term.vertices.length === 0 && !isDrawingTerm) return;
      const selected = state.selectedObject && state.selectedObject.type === 'terminal' && state.selectedObject.id === term.id;
      const buildingTheme = getBuildingTheme(term);
      const termPts = term.vertices.map(function(v) { return cellToPixel(v.col, v.row); });
      ctx.lineWidth = selected ? 3 : 2;
      ctx.strokeStyle = selected ? c2dObjectSelectedStroke() : buildingTheme.stroke;
      ctx.fillStyle = selected ? c2dObjectSelectedFill() : buildingTheme.fill;
      ctx.beginPath();
      for (let i = 0; i < termPts.length; i++) {
        const [x,y] = termPts[i];
        if (i === 0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
      }
      if (term.closed) {
        ctx.closePath();
        if (buildingTheme.fillEnabled) ctx.fill();
      }
      if (selected) {
        ctx.save();
        ctx.shadowColor = c2dObjectSelectedGlow();
        ctx.shadowBlur = c2dObjectSelectedGlowBlur();
        ctx.shadowOffsetX = 0;
        ctx.shadowOffsetY = 0;
      }
      ctx.stroke();
      if (selected) ctx.restore();
      if (term.closed && buildingTheme.hatch === 'diagonal' && buildingTheme.fillEnabled) {
        drawPolygonHatch(termPts, selected ? c2dObjectSelectedDashStroke() : buildingTheme.stroke, Math.max(10, CELL_SIZE * 0.6));
      }
      if (term.closed && term.vertices.length > 0) {
        let cx = 0, cy = 0;
        term.vertices.forEach(v => {
          const [px, py] = cellToPixel(v.col, v.row);
          cx += px; cy += py;
        });
        cx /= term.vertices.length;
        cy /= term.vertices.length;
        const label = term.name || term.id || 'Building';
        ctx.fillStyle = buildingTheme.labelFill;
        ctx.font = '12px system-ui';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(label, cx, cy);
      }
      term.vertices.forEach((v, i) => {
        const [x,y] = cellToPixel(v.col, v.row);
        const vertexSelected = isSelectedVertex('terminal', term.id, i);
        ctx.beginPath();
        ctx.fillStyle = vertexSelected ? '#f43f5e' : (i === 0 ? '#f97316' : '#e5e7eb');
        ctx.arc(x, y, layoutTerminalVertexRadiusPx(vertexSelected), 0, Math.PI*2);
        ctx.fill();
      });
      if (isDrawingTerm && state.layoutPathDrawPointer && term.vertices.length >= 1) {
        const ptr = state.layoutPathDrawPointer;
        const lastV = term.vertices[term.vertices.length - 1];
        const [lx, ly] = cellToPixel(lastV.col, lastV.row);
        if (ptr && ptr.length >= 2 && dist2([lx, ly], ptr) > 1e-6) {
          ctx.save();
          ctx.strokeStyle = 'rgba(250, 204, 21, 0.75)';
          ctx.setLineDash([4, 6]);
          ctx.lineWidth = 2;
          ctx.lineCap = 'round';
          ctx.beginPath();
          ctx.moveTo(lx, ly);
          ctx.lineTo(ptr[0], ptr[1]);
          ctx.stroke();
          ctx.restore();
        }
      }
    });
    ctx.restore();
  }

  function drawPBBs() {
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    state.pbbStands.forEach(pbb => {
      const x1 = Number(pbb.x1), y1 = Number(pbb.y1), x2 = Number(pbb.x2), y2 = Number(pbb.y2);
      if (!Number.isFinite(x1) || !Number.isFinite(y1) || !Number.isFinite(x2) || !Number.isFinite(y2)) return;
      rebuildPbbBridgeGeometry(pbb);
      const endSize = getStandSizeMeters(pbb.category || 'C');
      const sel = state.selectedObject && state.selectedObject.type === 'pbb' && state.selectedObject.id === pbb.id;
      const simOcc = state.hasSimulationResult && isStandOccupiedAtSimSec(pbb.id, state.simTimeSec);
      const bridges = Array.isArray(pbb.pbbBridges) ? pbb.pbbBridges : [];
      bridges.forEach(function(bridge, bridgeIdx) {
        const pts = Array.isArray(bridge.points) ? bridge.points : [];
        if (pts.length < 2) return;
        ctx.strokeStyle = sel ? c2dObjectSelectedStroke() : '#f97316';
        ctx.lineWidth = sel ? 3.5 : 2.5;
        if (sel) {
          ctx.save();
          ctx.shadowColor = c2dObjectSelectedGlow();
          ctx.shadowBlur = c2dObjectSelectedGlowBlur();
        }
        ctx.beginPath();
        ctx.moveTo(Number(pts[0].x) || 0, Number(pts[0].y) || 0);
        for (let pi = 1; pi < pts.length; pi++) ctx.lineTo(Number(pts[pi].x) || 0, Number(pts[pi].y) || 0);
        ctx.stroke();
        if (sel) ctx.restore();
        if (sel) {
          pts.forEach(function(pt, ptIdx) {
            const isBridgeVertexSelected = !!(state.selectedVertex && state.selectedVertex.type === 'pbbBridge' && state.selectedVertex.id === pbb.id && state.selectedVertex.bridgeIndex === bridgeIdx && state.selectedVertex.pointIndex === ptIdx);
            ctx.beginPath();
            ctx.fillStyle = isBridgeVertexSelected ? '#f43f5e' : '#fdba74';
            ctx.arc(Number(pt.x) || 0, Number(pt.y) || 0, isBridgeVertexSelected ? 4 : 3, 0, Math.PI * 2);
            ctx.fill();
          });
        }
      });
      const apronPt = getStandConnectionPx(pbb);
      const ex = apronPt[0], ey = apronPt[1];
      const angle = getPBBStandAngle(pbb);
      const rotationActive = !!(state.selectedVertex && state.selectedVertex.type === 'standRotation' && state.selectedVertex.id === pbb.id);
      ctx.fillStyle = sel ? c2dObjectSelectedFill() : (simOcc ? c2dSimStandOccupiedFill() : 'rgba(22,163,74,0.18)');
      ctx.strokeStyle = sel ? c2dObjectSelectedStroke() : (simOcc ? c2dSimStandOccupiedStroke() : '#22c55e');
      ctx.lineWidth = sel ? 2.5 : 1.5;
      ctx.save();
      ctx.translate(ex, ey);
      ctx.rotate(angle);
      ctx.beginPath();
      ctx.rect(-endSize/2, -endSize/2, endSize, endSize);
      ctx.fill();
      if (sel) {
        ctx.save();
        ctx.shadowColor = c2dObjectSelectedGlow();
        ctx.shadowBlur = c2dObjectSelectedGlowBlur();
      }
      ctx.stroke();
      if (sel) ctx.restore();
      const nameRaw = (pbb.name && pbb.name.trim()) ? pbb.name.trim() : String(state.pbbStands.indexOf(pbb) + 1);
      const labelPrefix = getStandCategoryMode(pbb) === 'aircraft' ? 'AC' : (pbb.category || 'C');
      const label = labelPrefix + ' / ' + nameRaw;
      const pad = 3;
      const tx = endSize / 2 - pad;
      const ty = -endSize / 2 + pad;
      ctx.fillStyle = '#bbf7d0';
      ctx.font = '8px system-ui';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'top';
      ctx.fillText(String(label), tx, ty);
      ctx.restore();
      ctx.save();
      ctx.beginPath();
      ctx.fillStyle = sel ? '#22c55e' : 'rgba(34,197,94,0.9)';
      ctx.arc(apronPt[0], apronPt[1], sel ? 4.5 : 3.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
      if (sel) {
        drawStandRotationHandle(getPbbRotationOriginPx(pbb), getPbbRotationHandlePx(pbb), rotationActive);
      }
    });
    ctx.restore();
  }

  function drawRemoteStands() {
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    const mode = settingModeSelect ? settingModeSelect.value : 'grid';
    state.remoteStands.forEach(st => {
      const [cx, cy] = getRemoteStandCenterPx(st);
      const size = getStandSizeMeters(st.category || 'C');
      const angle = getRemoteStandAngleRad(st);
      const sel = state.selectedObject && state.selectedObject.type === 'remote' && state.selectedObject.id === st.id;
      const simOcc = state.hasSimulationResult && isStandOccupiedAtSimSec(st.id, state.simTimeSec);
      const rotationActive = !!(state.selectedVertex && state.selectedVertex.type === 'standRotation' && state.selectedVertex.id === st.id);
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(angle);
      ctx.fillStyle = sel ? c2dObjectSelectedFill() : (simOcc ? c2dSimStandOccupiedFill() : 'rgba(22,163,74,0.18)');
      ctx.strokeStyle = sel ? c2dObjectSelectedStroke() : (simOcc ? c2dSimStandOccupiedStroke() : '#22c55e');
      ctx.lineWidth = sel ? 2.5 : 1.5;
      ctx.beginPath();
      ctx.rect(-size/2, -size/2, size, size);
      ctx.fill();
      if (sel) {
        ctx.save();
        ctx.shadowColor = c2dObjectSelectedGlow();
        ctx.shadowBlur = c2dObjectSelectedGlowBlur();
      }
      ctx.stroke();
      if (sel) ctx.restore();
      ctx.restore();
      if (mode === 'apronTaxiway') {
        ctx.save();
        ctx.fillStyle = sel ? '#f97316' : '#e5e7eb';
        ctx.beginPath();
        ctx.arc(cx, cy, 2.5 * LAYOUT_VERTEX_DOT_SCALE, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }
      const nameRaw = (st.name && st.name.trim()) ? st.name.trim() : ('R' + String(state.remoteStands.indexOf(st) + 1).padStart(3, '0'));
      const labelPrefix = getStandCategoryMode(st) === 'aircraft' ? 'AC' : (st.category || 'C');
      const label = labelPrefix + ' / ' + nameRaw;
      ctx.fillStyle = '#bbf7d0';
      ctx.font = '8px system-ui';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'top';
      const labelOffset = 2;
      ctx.fillText(label, cx + size/2 - labelOffset, cy - size/2 + labelOffset);
      if (sel) {
        drawStandRotationHandle([cx, cy], getRemoteRotationHandlePx(st), rotationActive);
      }
    });
    ctx.restore();
  }

  function renderRunwaySeparation() {
    const panel = document.getElementById('rwySepPanel');
    if (!panel) return;
    const runways = (state.taxiways || []).filter(tw => tw.pathType === 'runway');
    if (!runways.length) {
      panel.innerHTML = '<div style="font-size:11px;color:#9ca3af;">No runway paths. Layout Mode <strong>Runway</strong>Draw the runway polyline first with.</div>';
      return;
    }
    if (!state.activeRwySepId || !runways.some(r => r.id === state.activeRwySepId)) {
      state.activeRwySepId = runways[0].id;
    }
    const active = runways.find(r => r.id === state.activeRwySepId) || runways[0];
    const cfg = rsepGetConfigForRunway(active);
    const stdKey = cfg.standard || 'ICAO';
    const cats = RSEP_STD_CATS[stdKey] || RSEP_STD_CATS['ICAO'];
    const mode = cfg.mode || 'MIX';
    const seq = cfg.activeSeq || (RSEP_MODE_SEQS[mode] && RSEP_MODE_SEQS[mode][0]) || 'ARR→ARR';
    const seqType = RSEP_SEQ_TYPES[seq] || 'matrix';
    const seqMeta = rsepGetSeqMeta(seq);

    let html = '';
    html += '<div class="rwysep-rwy-bar">';
    html += '<div class="rwysep-rwy-tabs">';
    runways.forEach(rw => {
      const isActive = rw.id === active.id;
      const label = escapeHtml(rw.name || ('Runway ' + rw.id));
      html += '<button type="button" class="rwysep-rwy-btn' + (isActive ? ' active' : '') + '" data-rwy-id="' + rw.id + '">' + label + '</button>';
    });
    html += '</div></div>';

    const activeSub = 'noname';
    html += '<div class="layout-save-load-tabs" style="margin-top:8px;">';
    html += '<button type="button" class="layout-save-load-tab rwysep-subtab-btn active" data-subtab="noname">No Name</button>';
    html += '</div>';

    html += '<div id="rwysep-subtab-input" style="">';
    html += '<div class="rwysep-block">';
    html += '<div class="rwysep-label">Standard &amp; Mode</div>';
    html += '<div class="rwysep-row">';
    html += '<label style="font-size:11px;color:#9ca3af;">Standard&nbsp;</label>';
    html += '<select id="rwysep-standard">';
    html += '<option value="ICAO"' + (stdKey === 'ICAO' ? ' selected' : '') + '>ICAO (J/H/M/L)</option>';
    html += '<option value="RECAT-EU"' + (stdKey === 'RECAT-EU' ? ' selected' : '') + '>RECAT-EU (A~F)</option>';
    html += '</select>';
    html += '<label style="font-size:11px;color:#9ca3af;margin-left:8px;">Mode&nbsp;</label>';
    html += '<select id="rwysep-mode">';
    ['ARR','DEP','MIX'].forEach(m => {
      const txt = m === 'ARR' ? 'Arrivals only' : (m === 'DEP' ? 'Departures only' : 'Mixed (Arr/Dep)');
      html += '<option value="' + m + '"' + (mode === m ? ' selected' : '') + '>' + txt + '</option>';
    });
    html += '</select>';
    html += '<label style="font-size:11px;color:#9ca3af;margin-left:8px;">Seq&nbsp;</label>';
    html += '<select id="rwysep-seq">';
    (RSEP_MODE_SEQS[mode] || []).forEach(s => {
      const lbl = s;
      html += '<option value="' + s + '"' + (seq === s ? ' selected' : '') + '>' + lbl + '</option>';
    });
    html += '</select>';
    html += '</div></div>';

    if (seqMeta) {
      html += '<div class="rwysep-block" style="margin-top:4px;">';
      html += '<div class="rwysep-label">Concept summary</div>';
      html += '<div style="font-size:10px;color:#d1d5db;line-height:1.5;background:#020617;border-radius:6px;border:1px solid #111827;padding:6px 8px;">';
      html += '<div><span style="color:#9ca3af;">Driving factor</span>&nbsp;&nbsp;: ' + escapeHtml(seqMeta.driver) + '</div>';
      html += '<div><span style="color:#9ca3af;">Reference point</span>&nbsp;: ' + escapeHtml(seqMeta.refPoint) + '</div>';
      html += '<div><span style="color:#9ca3af;">Input structure</span>: ' + escapeHtml(seqMeta.input) + '</div>';
      html += '</div>';
      html += '</div>';
    }

    if (seq === 'ARR→DEP') {
      html += '<div class="rwysep-block">';
      html += '<div style="font-size:10px;color:#9ca3af;line-height:1.5;margin-bottom:6px;">Separation combines leading aircraft ROT with trailing aircraft lineup–gear-off time, using the ROT inputs above per wake category.</div>';

      const totalRot = cats.length;
      let filledRot = 0;
      cats.forEach(c => {
        const val = cfg.rot && cfg.rot[c] != null ? cfg.rot[c] : '';
        if (val !== '' && val != null) filledRot += 1;
      });
      html += rsepLegendHtml(filledRot, totalRot);

      html += '<div class="rwysep-row" style="flex-wrap:wrap;">';
      cats.forEach(c => {
        const rawVal = cfg.rot && cfg.rot[c] != null ? cfg.rot[c] : '';
        const valStr = rawVal === null || rawVal === undefined ? '' : String(rawVal);
        const sub = rsepGetCatLabel(stdKey, c);
        const colInfo = rsepColorForValue(valStr);
        html += '<div style="min-width:90px;margin-right:6px;margin-bottom:4px;">';
        html += '<div style="font-size:10px;color:#9ca3af;margin-bottom:2px;line-height:1.2;">';
        html += 'Cat ' + c;
        if (sub) {
          html += '<div style="font-size:9px;color:#6b7280;margin-top:1px;">' + escapeHtml(sub) + '</div>';
        }
        html += '</div>';
        html += '<input type="number" min="0" step="5" data-rwysep-rot="' + c + '" value="' + escapeHtml(valStr) + '" style="width:64px;background:' + colInfo.bg + ';border:1px solid ' + colInfo.border + ';color:' + colInfo.color + ';font-size:10px;padding:3px 4px;border-radius:3px;text-align:center;" />';
        html += ' <span style="font-size:9px;color:#6b7280;">sec</span>';
        html += '</div>';
      });
      html += '</div></div>';
    }

    if (seq !== 'ARR→DEP') {
      html += '<div class="rwysep-block">';
      html += '<div class="rwysep-label">Separation (sec) — ' + escapeHtml(seq) + '</div>';
      if (seqType === 'matrix') {
        const data = cfg.seqData && cfg.seqData[seq] ? cfg.seqData[seq] : rsepMakeMatrix(cats, null);
        const total = cats.length * cats.length;
        let filled = 0;
        cats.forEach(lead => {
          cats.forEach(trail => {
            const v = data[lead] && data[lead][trail] != null ? data[lead][trail] : '';
            if (v !== '' && v != null) filled += 1;
          });
        });
        html += rsepLegendHtml(filled, total);

        html += '<div class="rwysep-matrix-wrap"><table class="rwysep-table"><thead><tr>';
        html += '<th>Lead↓ / Trail→</th>';
        cats.forEach(c => {
          const sub = rsepGetCatLabel(stdKey, c);
          html += '<th><div style="line-height:1.2;">' + c;
          if (sub) {
            html += '<div style="font-size:9px;color:#9ca3af;margin-top:1px;">' + escapeHtml(sub) + '</div>';
          }
          html += '</div></th>';
        });
        html += '</tr></thead><tbody>';
        cats.forEach(lead => {
          const leadSub = rsepGetCatLabel(stdKey, lead);
          html += '<tr><td><div style="line-height:1.2;">' + lead;
          if (leadSub) {
            html += '<div style="font-size:9px;color:#9ca3af;margin-top:1px;">' + escapeHtml(leadSub) + '</div>';
          }
          html += '</div></td>';
          cats.forEach(trail => {
            const v = data[lead] && data[lead][trail] != null ? data[lead][trail] : '';
            const colInfo = rsepColorForValue(v);
            html += '<td><input type="number" min="0" step="5" data-rwysep-matrix-lead="' + lead + '" data-rwysep-matrix-trail="' + trail + '" value="' + escapeHtml(String(v)) + '" style="width:64px;background:' + colInfo.bg + ';border:1px solid ' + colInfo.border + ';color:' + colInfo.color + ';font-size:10px;padding:3px 4px;border-radius:3px;text-align:center;" /></td>';
          });
          html += '</tr>';
        });
        html += '</tbody></table></div>';
      } else {
        const data1d = cfg.seqData && cfg.seqData[seq] ? cfg.seqData[seq] : rsepMake1D(cats, null);
        const total = cats.length;
        let filled = 0;
        cats.forEach(cat => {
          const v = data1d[cat] != null ? data1d[cat] : '';
          if (v !== '' && v != null) filled += 1;
        });
        html += rsepLegendHtml(filled, total);

        html += '<div class="rwysep-row" style="flex-wrap:wrap;margin-top:4px;">';
        cats.forEach(cat => {
          const v = data1d[cat] != null ? data1d[cat] : '';
          const colInfo = rsepColorForValue(v);
          const sub = rsepGetCatLabel(stdKey, cat);
          html += '<div style="min-width:90px;margin-right:6px;margin-bottom:4px;border:1px solid #1f2937;border-radius:6px;padding:6px 8px;background:#020617;">';
          html += '<div style="font-size:10px;color:#9ca3af;margin-bottom:2px;line-height:1.2;">Cat ' + cat;
          if (sub) {
            html += '<div style="font-size:9px;color:#6b7280;margin-top:1px;">' + escapeHtml(sub) + '</div>';
          }
          html += '</div>';
          html += '<input type="number" min="0" step="5" data-rwysep-1d="' + cat + '" value="' + escapeHtml(String(v)) + '" style="width:64px;background:' + colInfo.bg + ';border:1px solid ' + colInfo.border + ';color:' + colInfo.color + ';font-size:10px;padding:3px 4px;border-radius:3px;text-align:center;" />';
          html += ' <span style="font-size:9px;color:#6b7280;">sec</span>';
          html += '</div>';
        });
        html += '</div>';
      }
      html += '</div>';
    }
    html += '</div>'; // end subtab input

    html += '<div id="rwysep-subtab-timeline" style="' + (activeSub === 'timeline' ? '' : 'display:none;') + '">';
    html += '<div class="rwysep-block" style="margin-top:8px;">';
    html += '<div class="rwysep-label">Separation Timeline (Reg × Time)</div>';
    html += '<div id="rwySepTimeWrap" style="width:100%;background:#020617;border-radius:8px;border:1px solid #1f2937;position:relative;overflow-x:auto;overflow-y:auto;margin-top:4px;max-height:calc(40px * 12 + 80px);"></div>';
    html += '<div style="font-size:9px;color:#9ca3af;margin-top:4px;">';
    html += 'Y: Reg Number · X: Time · Bars = S-series (SLDT–STOT) · Lines = E-series (ELDT–ETOT)';
    html += '</div></div>';
    html += '</div>'; // end subtab timeline

    panel.innerHTML = html;

    function drawRwySeparationTimeline() {
      if (state.activeRwySepSubtab && state.activeRwySepSubtab !== 'timeline') return;
      const wrap = panel.querySelector('#rwySepTimeWrap');
      if (!wrap) return;

      const allData = typeof buildRunwaySeparationTimelineByRunwaySnapshot === 'function'
        ? buildRunwaySeparationTimelineByRunwaySnapshot(state.flights)
        : null;
      const data = allData && active && active.id != null ? allData[active.id] : null;
      if (!data || !data.events || !data.events.length) {
        wrap.innerHTML = '<div style="font-size:11px;color:#9ca3af;padding:8px 10px;">No SLDT/STOT events for this runway.</div>';
        return;
      }

      const byFlight = new Map();
      data.events.forEach(ev => {
        const f = ev.flight;
        if (!f) return;
        let lane = byFlight.get(f);
        if (!lane) {
          const reg = f.reg || f.id || '';
          lane = {
            flight: f,
            reg,
            hasArr: false,
            hasDep: false,
            sldt: null,
            eldt: null,
            stot: null,
            etot: null
          };
          byFlight.set(f, lane);
        }
        if (ev.type === 'arr') {
          lane.hasArr = true;
          lane.sldt = ev.time;
          lane.eldt = (f.eldtMin != null ? f.eldtMin : ev.time);
        } else if (ev.type === 'dep') {
          lane.hasDep = true;
          lane.stot = ev.time;
          lane.etot = (f.etotMin != null ? f.etotMin : ev.time);
        }
      });

      const lanes = Array.from(byFlight.values());
      if (!lanes.length) {
        wrap.innerHTML = '<div style="font-size:11px;color:#9ca3af;padding:8px 10px;">No SLDT/STOT events for this runway.</div>';
        return;
      }

      let minT0 = Infinity;
      let maxT0 = -Infinity;
      lanes.forEach(ln => {
        if (ln.sldt != null && ln.sldt < minT0) minT0 = ln.sldt;
        if (ln.etot != null && ln.etot > maxT0) maxT0 = ln.etot;
      });
      if (minT0 <= 0 && lanes.length) {
        const pos = lanes.map(function(ln) { return ln.sldt; }).filter(function(v) { return v != null && isFinite(v) && v > 1e-6; });
        if (pos.length) minT0 = Math.min.apply(null, pos);
      }
      if (!isFinite(minT0) || !isFinite(maxT0)) {
        minT0 = data.minT;
        maxT0 = data.maxT;
      }
      let baseMinT = Math.max(0, minT0 - RWY_SEP_TIMELINE_PAD_MIN);
      let baseMaxT = maxT0 + RWY_SEP_TIMELINE_PAD_MIN;
      if (baseMaxT <= baseMinT) baseMaxT = baseMinT + 60;
      const baseSpan = baseMaxT - baseMinT;
      const zoom = (state.rwySepTimeZoom && state.rwySepTimeZoom > 1) ? state.rwySepTimeZoom : 1;
      const span = baseSpan;
      const minT = baseMinT;
      const maxT = baseMaxT;

      lanes.sort((a, b) => {
        const ta = (a.sldt ?? a.stot ?? a.eldt ?? a.etot ?? 0);
        const tb = (b.sldt ?? b.stot ?? b.eldt ?? b.etot ?? 0);
        return ta - tb;
      });

      const tickPositions = buildTimeAxisTicks(minT, maxT, baseMinT, baseSpan, zoom);

      const sMarkers = [];
      const eMarkers = [];

      const rows = [];
      lanes.forEach(ln => {
        const reg = ln.reg || '';
        const sStart = (ln.sldt != null ? ln.sldt : null);
        const sEnd = (ln.stot != null ? ln.stot : null);
        const eStart = (ln.eldt != null ? ln.eldt : null);
        const eEnd = (ln.etot != null ? ln.etot : null);

        let blocks = '';
        if (sStart != null && sEnd != null && span > 0) {
          const s1 = Math.max(sStart, baseMinT);
          const s2 = Math.min(sEnd, baseMaxT);
          if (s2 <= s1) return;
          const leftPct = ((s1 - baseMinT) / baseSpan) * 100 * zoom;
          const widthPct = Math.max(1, ((s2 - s1) / baseSpan) * 100 * zoom);
          const rightPct = leftPct + widthPct;
          sMarkers.push({ t: sStart, leftPct, type: 'start' });
          sMarkers.push({ t: sEnd, leftPct: rightPct, type: 'end' });
          blocks +=
            '<div class="rwysep-line-s" style="' +
              'left:' + leftPct + '%;' +
              'width:' + widthPct + '%;' +
            '"></div>' +
            '<div class="rwysep-tri" style="' +
              'top:20%;' +
              'left:' + leftPct + '%;' +
              'border-top:6px solid ' + GANTT_COLORS.S_SERIES + ';' +
            '"></div>' +
            '<div class="rwysep-tri" style="' +
              'top:20%;' +
              'left:' + rightPct + '%;' +
              'border-bottom:6px solid ' + GANTT_COLORS.S_SERIES + ';' +
            '"></div>';
        }
        if (eStart != null && eEnd != null && span > 0) {
          const e1 = Math.max(eStart, baseMinT);
          const e2 = Math.min(eEnd, baseMaxT);
          if (e2 <= e1) return;
          const leftPct2 = ((e1 - baseMinT) / baseSpan) * 100 * zoom;
          const widthPct2 = Math.max(0.5, ((e2 - e1) / baseSpan) * 100 * zoom);
          const rightPct2 = leftPct2 + widthPct2;
          eMarkers.push({ t: eStart, leftPct: leftPct2, type: 'start' });
          eMarkers.push({ t: eEnd, leftPct: rightPct2, type: 'end' });
          blocks +=
            '<div class="rwysep-line-e" style="' +
              'left:' + leftPct2 + '%;' +
              'width:' + widthPct2 + '%;' +
            '"></div>' +
            '<div class="rwysep-tri" style="' +
              'top:54%;' +
              'left:' + leftPct2 + '%;' +
              'border-top:6px solid ' + GANTT_COLORS.E_SERIES + ';' +
            '"></div>' +
            '<div class="rwysep-tri" style="' +
              'top:54%;' +
              'left:' + rightPct2 + '%;' +
              'border-bottom:6px solid ' + GANTT_COLORS.E_SERIES + ';' +
            '"></div>';
        }

        rows.push(
          '<div class="alloc-row">' +
            '<div class="alloc-row-label">' + escapeHtml(reg) + '</div>' +
            '<div class="alloc-row-track" style="background:transparent;border:none;">' + blocks + '</div>' +
          '</div>'
        );
      });

      sMarkers.sort((a, b) => a.t - b.t);
      eMarkers.sort((a, b) => a.t - b.t);

      const sHeadMarks = sMarkers.map(m =>
        '<div class="rwysep-tri" style="' +
          'top:60%;' +
          'left:' + m.leftPct + '%;' +
          (m.type === 'start'
            ? 'border-top:6px solid ' + GANTT_COLORS.S_SERIES + ';'
            : 'border-bottom:6px solid ' + GANTT_COLORS.S_SERIES + ';') +
        '"></div>'
      ).join('');

      const eHeadMarks = eMarkers.map(m =>
        '<div class="rwysep-tri" style="' +
          'top:60%;' +
          'left:' + m.leftPct + '%;' +
          (m.type === 'start'
            ? 'border-top:6px solid ' + GANTT_COLORS.E_SERIES + ';'
            : 'border-bottom:6px solid ' + GANTT_COLORS.E_SERIES + ';') +
        '"></div>'
      ).join('');

      const headHtml =
        '<div class="rwysep-head-row">' +
          '<div class="rwysep-head-label">S-series</div>' +
          '<div class="rwysep-head-track">' + sHeadMarks + '</div>' +
        '</div>' +
        '<div class="rwysep-head-row">' +
          '<div class="rwysep-head-label">E-series</div>' +
          '<div class="rwysep-head-track">' + eHeadMarks + '</div>' +
        '</div>';

      const axisTicks = tickPositions.map(tp =>
        '<div class="alloc-time-tick" style="left:' + tp.leftPct + '%;">' +
          '<div class="alloc-time-tick-label">' + tp.label + '</div>' +
        '</div>'
      );
      const axisHtml =
        '<div class="alloc-time-axis-overlay">' +
          '<div class="alloc-time-axis-inner">' + axisTicks.join('') + '</div>' +
        '</div>';

      const rwyGridOverlay =
        '<div class="alloc-gantt-grid-overlay">' +
          tickPositions.map(function(tp) {
            return '<div class="alloc-time-grid-line" style="left:' + tp.leftPct + '%;"></div>';
          }).join('') +
        '</div>';
      const rowsHtml = '<div class="rwysep-rows">' + rwyGridOverlay + rows.join('') + '</div>';
      wrap.innerHTML = headHtml + rowsHtml + axisHtml;

      if (!wrap._rwySepZoomBound) {
        wrap._rwySepZoomBound = true;
        wrap.addEventListener('wheel', function(e) {
          if (!e.shiftKey) return;
          e.preventDefault();
          const factor = e.deltaY < 0 ? 1.15 : (1 / 1.15);
          let z = state.rwySepTimeZoom || 1;
          z *= factor;
          if (z < 1) z = 1;
          if (z > 8) z = 8;
          state.rwySepTimeZoom = z;
          if (typeof renderRunwaySeparation === 'function') renderRunwaySeparation();
        }, { passive: false });
      }

      if (!wrap._rwySepScrollBound) {
        wrap._rwySepScrollBound = true;
        wrap.addEventListener('scroll', function() {
          if (wrap._rwySepScrollRecalc) return;
          const currentLeft = wrap.scrollLeft;
          wrap._rwySepScrollRecalc = true;
          drawRwySeparationTimeline();
          wrap.scrollLeft = currentLeft;
          wrap._rwySepScrollRecalc = false;
        });
      }
    }

    drawRwySeparationTimeline();

    _rwySepWireInputHandlers(panel, cfg, cats, seq, state);
  }

  function _rwySepWireInputHandlers(panel, cfg, cats, seq, st) {
    panel.querySelectorAll('.rwysep-rwy-btn').forEach(function(btn) {
      btn.addEventListener('click', function() {
        const id = this.getAttribute('data-rwy-id');
        if (!id) return;
        st.activeRwySepId = id;
        renderRunwaySeparation();
      });
    });
    panel.querySelectorAll('.rwysep-subtab-btn').forEach(function(btn) {
      btn.addEventListener('click', function() {
        const sub = this.getAttribute('data-subtab') || 'input';
        st.activeRwySepSubtab = sub;
        renderRunwaySeparation();
      });
    });
    var stdSel = panel.querySelector('#rwysep-standard');
    if (stdSel) {
      stdSel.addEventListener('change', function() {
        cfg.standard = this.value || 'ICAO';
        cfg.seqData = rsepMakeSeqData(cfg.standard);
        var catsNew = RSEP_STD_CATS[cfg.standard] || [];
        var rotNew = RSEP_STANDARDS[cfg.standard] && RSEP_STANDARDS[cfg.standard].ROT || {};
        cfg.rot = {};
        catsNew.forEach(function(c) { cfg.rot[c] = rotNew[c] != null ? String(rotNew[c]) : ''; });
        renderRunwaySeparation();
      });
    }
    var modeSel = panel.querySelector('#rwysep-mode');
    if (modeSel) {
      modeSel.addEventListener('change', function() {
        cfg.mode = this.value || 'MIX';
        var seqs = RSEP_MODE_SEQS[cfg.mode] || ['ARR→ARR'];
        if (!seqs.includes(cfg.activeSeq)) cfg.activeSeq = seqs[0];
        renderRunwaySeparation();
      });
    }
    var seqSel = panel.querySelector('#rwysep-seq');
    if (seqSel) {
      seqSel.addEventListener('change', function() {
        cfg.activeSeq = this.value || 'ARR→ARR';
        renderRunwaySeparation();
      });
    }
    function _applyColorOnChange(inp) {
      var colInfo = rsepColorForValue(inp.value);
      inp.style.background = colInfo.bg;
      inp.style.borderColor = colInfo.border;
      inp.style.color = colInfo.color;
    }
    panel.querySelectorAll('input[data-rwysep-rot]').forEach(function(inp) {
      inp.addEventListener('change', function() {
        var cat = this.getAttribute('data-rwysep-rot');
        if (!cat) return;
        cfg.rot[cat] = this.value;
        _applyColorOnChange(this);
      });
    });
    panel.querySelectorAll('input[data-rwysep-matrix-lead]').forEach(function(inp) {
      inp.addEventListener('change', function() {
        var lead = this.getAttribute('data-rwysep-matrix-lead');
        var trail = this.getAttribute('data-rwysep-matrix-trail');
        if (!lead || !trail) return;
        if (!cfg.seqData[seq]) cfg.seqData[seq] = rsepMakeMatrix(cats, null);
        if (!cfg.seqData[seq][lead]) cfg.seqData[seq][lead] = {};
        cfg.seqData[seq][lead][trail] = this.value;
        _applyColorOnChange(this);
      });
    });
    panel.querySelectorAll('input[data-rwysep-1d]').forEach(function(inp) {
      inp.addEventListener('change', function() {
        var cat = this.getAttribute('data-rwysep-1d');
        if (!cat) return;
        if (!cfg.seqData[seq]) cfg.seqData[seq] = rsepMake1D(cats, null);
        cfg.seqData[seq][cat] = this.value;
        _applyColorOnChange(this);
      });
    });
  }

  function taxiwayPathWidthBodyAlpha(tw, sel, drawing) {
    if (sel || drawing || state.showRoadWidth) return 1;
    if (tw.pathType === 'runway') {
      const n = Number(_canvas2dStyle.runwayPathWidthOffAlpha);
      return (isFinite(n) && n >= 0 && n <= 1) ? n : 0.3;
    }
    return 0;
  }
  function drawTaxiways() {
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    state.taxiways.forEach(tw => {
      const drawing = state.taxiwayDrawingId === tw.id;
      if (tw.vertices.length < 2 && !drawing) return;
      const isRunwayPath = tw.pathType === 'runway';
      const isRunwayExit = tw.pathType === 'runway_exit';
      const widthDefault = isRunwayPath ? RUNWAY_PATH_DEFAULT_WIDTH : (isRunwayExit ? RUNWAY_EXIT_DEFAULT_WIDTH : TAXIWAY_DEFAULT_WIDTH);
      const width = tw.width != null ? tw.width : widthDefault;
      const sel = state.selectedObject && state.selectedObject.type === 'taxiway' && state.selectedObject.id === tw.id;
      const pathLineCap = 'butt';
      const pathBodyAlpha = taxiwayPathWidthBodyAlpha(tw, sel, drawing);
      if (sel) {
        ctx.strokeStyle = c2dObjectSelectedStroke();
        ctx.fillStyle = c2dObjectSelectedFill();
      } else if (isRunwayPath || isRunwayExit) {
        ctx.strokeStyle = c2dRunwayStroke();
        ctx.fillStyle = c2dRunwayFill();
      } else {
        ctx.strokeStyle = drawing ? 'rgba(56, 189, 248, 0.72)' : c2dRunwayStroke();
        ctx.fillStyle = drawing ? 'rgba(56, 189, 248, 0.14)' : c2dRunwayFill();
      }
      ctx.lineWidth = width;
      ctx.lineCap = pathLineCap;
      ctx.lineJoin = 'round';
      ctx.beginPath();
      for (let i = 0; i < tw.vertices.length; i++) {
        const [x, y] = cellToPixel(tw.vertices[i].col, tw.vertices[i].row);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      if (tw.vertices.length >= 2) {
        ctx.save();
        ctx.globalAlpha = pathBodyAlpha;
        if (sel) {
          ctx.shadowColor = c2dObjectSelectedGlow();
          ctx.shadowBlur = c2dObjectSelectedGlowBlur();
          ctx.stroke();
        } else ctx.stroke();
        ctx.restore();
      }
      if (!isRunwayPath) {
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = sel ? c2dObjectSelectedStroke() : (isRunwayExit ? c2dRunwayTaxiwayCenterlineStroke() : c2dTaxiwayCenterlineStroke());
        ctx.beginPath();
        for (let i = 0; i < tw.vertices.length; i++) {
          const [x, y] = cellToPixel(tw.vertices[i].col, tw.vertices[i].row);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        if (tw.vertices.length >= 2) ctx.stroke();
      }
      if (isRunwayPath && tw.vertices.length >= 2) {
        const runwayPts = tw.vertices.map(v => cellToPixel(v.col, v.row));
        ctx.save();
        ctx.globalAlpha = pathBodyAlpha;
        drawRunwayDecorations(tw, runwayPts, width);
        ctx.restore();
      }
      const dir = getTaxiwayDirection(tw);
      if (dir !== 'both' && tw.vertices.length >= 2) {
        const pts = tw.vertices.map(v => cellToPixel(v.col, v.row));
        const totalLen = pts.reduce((acc, p, i) => acc + (i > 0 ? Math.hypot(p[0]-pts[i-1][0], p[1]-pts[i-1][1]) : 0), 0);
        const arrowSpacing = Math.max(22, Math.min(42, totalLen / 10));
        const numArrows = Math.max(2, Math.floor(totalLen / arrowSpacing));
        const arrLen = CELL_SIZE * 0.54;
        ctx.fillStyle = isRunwayPath ? '#f5930b' : (isRunwayExit ? c2dRunwayTaxiwayCenterlineStroke() : c2dTaxiwayCenterlineStroke());
        for (let k = 1; k <= numArrows; k++) {
          const targetDist = totalLen * (k / (numArrows + 1));
          let acc = 0;
          let ax = pts[0][0], ay = pts[0][1];
          let angle = Math.atan2(pts[1][1]-pts[0][1], pts[1][0]-pts[0][0]);
          for (let i = 1; i < pts.length; i++) {
            const seg = Math.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1]);
            angle = Math.atan2(pts[i][1]-pts[i-1][1], pts[i][0]-pts[i-1][0]);
            if (acc + seg >= targetDist) {
              const t = seg > 0 ? (targetDist - acc) / seg : 0;
              ax = pts[i-1][0] + t * (pts[i][0]-pts[i-1][0]);
              ay = pts[i-1][1] + t * (pts[i][1]-pts[i-1][1]);
              break;
            }
            acc += seg;
          }
          if (dir === 'counter_clockwise') angle += Math.PI;
          ctx.beginPath();
          ctx.moveTo(ax + arrLen * Math.cos(angle), ay + arrLen * Math.sin(angle));
          ctx.lineTo(ax - arrLen * 0.7 * Math.cos(angle) + arrLen * 0.4 * Math.sin(angle), ay - arrLen * 0.7 * Math.sin(angle) - arrLen * 0.4 * Math.cos(angle));
          ctx.lineTo(ax - arrLen * 0.7 * Math.cos(angle) - arrLen * 0.4 * Math.sin(angle), ay - arrLen * 0.7 * Math.sin(angle) + arrLen * 0.4 * Math.cos(angle));
          ctx.closePath();
          ctx.fill();
        }
      }
      if (isRunwayPath && tw.vertices.length >= 2) {
        const rp = getRunwayPath(tw.id);
        if (rp && rp.pts.length >= 2) {
          const lenPx = runwayPolylineLengthPx(rp.pts);
          const d = Math.min(Math.max(0, getEffectiveRunwayLineupDistM(tw)), lenPx);
          const lp = getRunwayPointAtDistance(tw.id, d);
          if (lp) {
            const lineupRtxOk = isLineupPointTouchingRunwayTaxiwayOnRunway(tw, lp);
            ctx.save();
            ctx.fillStyle = lineupRtxOk ? '#16a34a' : '#dc2626';
            ctx.strokeStyle = lineupRtxOk ? '#14532d' : '#450a0a';
            ctx.lineWidth = 1.2;
            ctx.beginPath();
            ctx.arc(lp[0], lp[1], 5 * LAYOUT_VERTEX_DOT_SCALE, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
            const labelText = 'Line up';
            ctx.font = 'bold 11px system-ui, sans-serif';
            const padX = 6, padY = 4, rad = 5;
            const mLabel = ctx.measureText(labelText);
            const bw = mLabel.width + padX * 2;
            const bh = 11 + padY * 2;
            const bx = lp[0] + 7;
            const by = lp[1] - 4 - bh;
            ctx.beginPath();
            if (typeof ctx.roundRect === 'function') ctx.roundRect(bx, by, bw, bh, rad);
            else ctx.rect(bx, by, bw, bh);
            ctx.fillStyle = lineupRtxOk ? 'rgba(22, 163, 74, 0.92)' : 'rgba(220, 38, 38, 0.92)';
            ctx.fill();
            ctx.strokeStyle = lineupRtxOk ? '#14532d' : '#450a0a';
            ctx.lineWidth = 1.2;
            ctx.stroke();
            ctx.fillStyle = '#ffffff';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(labelText, bx + bw / 2, by + bh / 2);
            ctx.restore();
            const hop1 = listRtxTouchingLineupOnRunway(tw, lp);
            for (let hi = 0; hi < hop1.length; hi++) {
              const rtx = hop1[hi];
              if (!rtx) continue;
              const nid = rtxRunwayExitNeighborIds(rtx);
              if (typeof rtxSetHasRunwayHoldingHp === 'function' && rtxSetHasRunwayHoldingHp(nid)) continue;
              const vts = rtx.vertices || [];
              if (vts.length < 2) continue;
              let sx = 0, sy = 0;
              for (let vi = 0; vi < vts.length; vi++) {
                const pp = cellToPixel(vts[vi].col, vts[vi].row);
                sx += pp[0]; sy += pp[1];
              }
              const mx = sx / vts.length, my = sy / vts.length;
              const badgeText = 'No Holding Point';
              ctx.save();
              ctx.font = 'bold 10px system-ui, sans-serif';
              const padXB = 6, padYB = 3, radB = 4;
              const mw = ctx.measureText(badgeText).width + padXB * 2;
              const mh = 10 + padYB * 2;
              const bxx = mx - mw / 2, byy = my - 22;
              ctx.beginPath();
              if (typeof ctx.roundRect === 'function') ctx.roundRect(bxx, byy, mw, mh, radB);
              else ctx.rect(bxx, byy, mw, mh);
              ctx.fillStyle = 'rgba(220, 38, 38, 0.95)';
              ctx.fill();
              ctx.strokeStyle = '#450a0a';
              ctx.lineWidth = 1.1;
              ctx.stroke();
              ctx.fillStyle = '#ffffff';
              ctx.textAlign = 'center';
              ctx.textBaseline = 'middle';
              ctx.fillText(badgeText, bxx + mw / 2, byy + mh / 2);
              ctx.restore();
            }
          }
        }
      }
      if ((drawing || sel) && tw.vertices.length >= 1) {
        tw.vertices.forEach((v, i) => {
          const [x, y] = cellToPixel(v.col, v.row);
          const vertexSelected = isSelectedVertex('taxiway', tw.id, i);
          if (i === 0 && drawing) {
            ctx.fillStyle = '#f97316';
            ctx.beginPath();
            ctx.arc(x, y, c2dPathDrawStartMarkerRadiusPx(), 0, Math.PI*2);
            ctx.fill();
            ctx.strokeStyle = '#ea580c';
            ctx.lineWidth = c2dPathDrawStartMarkerStrokePx();
            ctx.stroke();
            ctx.fillStyle = '#fff';
            ctx.font = 'bold ' + c2dPathDrawStartLabelFontPx() + 'px system-ui';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('Start', x, y + c2dPathDrawStartLabelOffsetY());
          } else {
            ctx.fillStyle = vertexSelected ? '#f43f5e' : ((i === 0 && sel) ? '#f97316' : '#e5e7eb');
            ctx.beginPath();
            ctx.arc(x, y, layoutPathVertexRadiusPx(vertexSelected, sel), 0, Math.PI*2);
            ctx.fill();
          }
        });
      }
      if (drawing && state.layoutPathDrawPointer && tw.vertices.length >= 1) {
        const ptr = state.layoutPathDrawPointer;
        const lastV = tw.vertices[tw.vertices.length - 1];
        const [lx, ly] = cellToPixel(lastV.col, lastV.row);
        if (ptr && ptr.length >= 2 && dist2([lx, ly], ptr) > 1e-6) {
          ctx.save();
          ctx.strokeStyle = 'rgba(56, 189, 248, 0.75)';
          ctx.setLineDash([4, 6]);
          ctx.lineWidth = Math.max(2, width * 0.25);
          ctx.lineCap = 'round';
          ctx.beginPath();
          ctx.moveTo(lx, ly);
          ctx.lineTo(ptr[0], ptr[1]);
          ctx.stroke();
          ctx.restore();
        }
      }
    });
    ctx.restore();
  }

  function drawApronTaxiwayLinks() {
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.scale, state.scale);
    ctx.lineWidth = 3;
    ctx.setLineDash([3, 3]);
    state.apronLinks.forEach(lk => {
      const stand = findStandById(lk.pbbId);
      const tw = state.taxiways.find(t => t.id === lk.taxiwayId);
      if (!stand || !tw || lk.tx == null || lk.ty == null) return;
      const poly = getApronLinkPolylineWorldPts(lk);
      if (poly.length < 2) return;
      ctx.strokeStyle = '#facc15';
      ctx.beginPath();
      ctx.moveTo(poly[0][0], poly[0][1]);
      for (let pi = 1; pi < poly.length; pi++) ctx.lineTo(poly[pi][0], poly[pi][1]);
      ctx.stroke();
      const svApron = state.selectedVertex;
      const selApron = state.selectedObject && state.selectedObject.type === 'apronLink' && state.selectedObject.id === lk.id;
      if (selApron) {
        ctx.setLineDash([]);
        for (let pi = 0; pi < poly.length; pi++) {
          const [px, py] = poly[pi];
          const isStandEnd = (pi === 0);
          const isTaxiEnd = (pi === poly.length - 1);
          const midIdx = isStandEnd || isTaxiEnd ? -1 : (pi - 1);
          let vtxSel = false;
          let draggable = false;
          if (isTaxiEnd) {
            draggable = true;
            vtxSel = !!(svApron && svApron.type === 'apronLink' && svApron.id === lk.id && svApron.kind === 'taxiway');
          } else if (!isStandEnd) {
            draggable = true;
            vtxSel = !!(svApron && svApron.type === 'apronLink' && svApron.id === lk.id && svApron.kind === 'mid' && svApron.midIndex === midIdx);
          }
          const r = layoutPathVertexRadiusPx(vtxSel, draggable);
          ctx.fillStyle = vtxSel ? '#f43f5e' : (draggable ? '#fde68a' : '#facc15');
          ctx.beginPath();
          ctx.arc(px, py, r, 0, Math.PI*2);
          ctx.fill();
        }
        ctx.setLineDash([3, 3]);
      }
    });
    ctx.setLineDash([]);
    if (state.apronLinkTemp) {
      ctx.fillStyle = '#facc15';
      const t = state.apronLinkTemp;
      const draft = [];
      if (t.kind === 'pbb' || t.kind === 'remote') {
        const st = findStandById(t.standId);
        if (st) {
          draft.push(getStandConnectionPx(st));
        }
      } else if (t.kind === 'taxiway') {
        draft.push([t.x, t.y]);
      }
      (state.apronLinkMidpoints || []).forEach(function(c) {
        draft.push(cellToPixel(c.col, c.row));
      });
      if (state.apronLinkPointerWorld && state.apronLinkPointerWorld.length >= 2) draft.push(state.apronLinkPointerWorld);
      if (draft.length >= 1) {
        ctx.save();
        ctx.strokeStyle = 'rgba(250, 204, 21, 0.75)';
        ctx.setLineDash([4, 6]);
