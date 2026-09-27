/*
 * NeuroHand 3D prosthetic hand (three.js r128, loaded as a global).
 *
 * A right hand seen from the palm side: palm shell, four fingers with three
 * phalanges each, an opposable thumb and a wrist socket with a status ring.
 * Every joint is a pivot group; a pose is a set of target joint angles that the
 * hand eases towards, like servos pulling tendons.
 *
 *   var hand = Hand3D.create(containerElement);   // null if WebGL is missing
 *   hand.setPose('fist', '#C0432B');               // open | fist | pinch | hold
 */
(function (root) {
  'use strict';

  // Joint angles in radians. Fingers: [spread, MCP, PIP, DIP].
  // Thumb: [abduction, opposition, MCP, IP].
  var POSES = {
    open: {
      index: [-0.12, 0.04, 0.04, 0.02], middle: [-0.03, 0.04, 0.04, 0.02],
      ring: [0.07, 0.04, 0.04, 0.02], little: [0.17, 0.04, 0.04, 0.02],
      thumb: [-0.85, 0.05, 0.05, 0.05]
    },
    fist: {
      index: [0.02, 1.5, 1.75, 1.05], middle: [0, 1.5, 1.75, 1.05],
      ring: [-0.02, 1.5, 1.75, 1.05], little: [-0.05, 1.45, 1.7, 1.0],
      thumb: [-0.25, 1.05, 0.55, 0.75]
    },
    // Thumb and index tips meet (tip centres ~1.15 cm apart = touching pads)
    pinch: {
      index: [0.02, 0.8, 1.2, 0.7], middle: [-0.02, 0.35, 0.45, 0.3],
      ring: [0.05, 0.4, 0.5, 0.32], little: [0.12, 0.45, 0.55, 0.35],
      thumb: [0, 0.9, 0.2, 0]
    },
    hold: {
      index: [-0.06, 0.3, 0.42, 0.25], middle: [-0.02, 0.32, 0.45, 0.27],
      ring: [0.04, 0.34, 0.47, 0.28], little: [0.1, 0.36, 0.5, 0.3],
      thumb: [-0.6, 0.35, 0.2, 0.2]
    }
  };

  // Finger layout on the palm (units ≈ cm). Palm faces +z (the camera).
  var FINGERS = {
    index: { x: 2.95, y: 9.0, r: 0.82, len: [3.9, 2.4, 1.9] },
    middle: { x: 0.98, y: 9.35, r: 0.86, len: [4.3, 2.8, 2.0] },
    ring: { x: -0.98, y: 9.1, r: 0.82, len: [4.0, 2.6, 1.9] },
    little: { x: -2.85, y: 8.5, r: 0.7, len: [3.2, 2.0, 1.7] }
  };

  function create(container) {
    var THREE = root.THREE;
    if (!THREE || !container) return null;
    var renderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    } catch (e) {
      return null;
    }
    if (!renderer.getContext()) return null;

    var reduceMotion = root.matchMedia && root.matchMedia('(prefers-reduced-motion: reduce)').matches;
    renderer.setPixelRatio(Math.min(root.devicePixelRatio || 1, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.outputEncoding = THREE.sRGBEncoding;
    renderer.domElement.setAttribute('aria-hidden', 'true');
    container.appendChild(renderer.domElement);

    var scene = new THREE.Scene();
    var camera = new THREE.PerspectiveCamera(30, 1, 0.1, 200);
    camera.position.set(12, 12, 44);
    var target = new THREE.Vector3(0, 6.8, 0);
    camera.lookAt(target);

    // Light: soft sky + key light with shadows + cool rim light
    scene.add(new THREE.HemisphereLight(0xf2f6f8, 0x5a6870, 0.85));
    var key = new THREE.DirectionalLight(0xffffff, 0.95);
    key.position.set(8, 20, 14);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    key.shadow.camera.left = -12; key.shadow.camera.right = 12;
    key.shadow.camera.top = 22; key.shadow.camera.bottom = -6;
    key.shadow.radius = 4;
    scene.add(key);
    var rim = new THREE.DirectionalLight(0x9fd8ea, 0.55);
    rim.position.set(-12, 10, -10);
    scene.add(rim);

    // Materials
    var shell = new THREE.MeshStandardMaterial({ color: 0xe8eef0, roughness: 0.42, metalness: 0.08 });
    var joint = new THREE.MeshStandardMaterial({ color: 0x22313a, roughness: 0.35, metalness: 0.65 });
    var carbon = new THREE.MeshStandardMaterial({ color: 0x2b3940, roughness: 0.55, metalness: 0.35 });
    var accent = new THREE.MeshStandardMaterial({ color: 0x8397a0, emissive: 0x000000, roughness: 0.3, metalness: 0.2 });

    function mesh(geometry, material) {
      var m = new THREE.Mesh(geometry, material);
      m.castShadow = true;
      m.receiveShadow = true;
      return m;
    }

    var hand = new THREE.Group();
    scene.add(hand);

    // Palm: rounded slab
    var shape = new THREE.Shape();
    var w = 4.3, h0 = 0.6, h1 = 9.2, rr = 1.6;
    shape.moveTo(-w + rr, h0);
    shape.lineTo(w - rr, h0);
    shape.quadraticCurveTo(w, h0, w, h0 + rr);
    shape.lineTo(w, h1 - rr);
    shape.quadraticCurveTo(w, h1, w - rr, h1);
    shape.lineTo(-w + rr, h1);
    shape.quadraticCurveTo(-w, h1, -w, h1 - rr);
    shape.lineTo(-w, h0 + rr);
    shape.quadraticCurveTo(-w, h0, -w + rr, h0);
    var palmGeo = new THREE.ExtrudeGeometry(shape, {
      depth: 1.9, bevelEnabled: true, bevelThickness: 0.55, bevelSize: 0.5, bevelSegments: 5, curveSegments: 14
    });
    palmGeo.translate(0, 0, -0.95);
    hand.add(mesh(palmGeo, shell));


    // Wrist socket with a status ring
    var socket = mesh(new THREE.CylinderGeometry(3.3, 3.6, 5.2, 48), carbon);
    socket.position.set(0, -2.1, 0);
    socket.scale.set(1.08, 1, 0.62);
    hand.add(socket);
    var ring = mesh(new THREE.TorusGeometry(3.45, 0.18, 12, 64), accent);
    ring.rotation.x = Math.PI / 2;
    ring.scale.set(1.08, 0.62, 1);
    ring.position.set(0, 0.1, 0);
    hand.add(ring);

    // Chain of phalanges: returns the three joint pivots
    function buildDigit(radius, lengths, parent) {
      var pivots = [], p = parent, r = radius;
      for (var s = 0; s < lengths.length; s++) {
        var len = lengths[s];
        var pivot = new THREE.Group();
        if (s > 0) pivot.position.y = lengths[s - 1];
        p.add(pivot);
        var knuckle = mesh(new THREE.SphereGeometry(r * 1.02, 24, 16), joint);
        pivot.add(knuckle);
        var bone = mesh(new THREE.CylinderGeometry(r * 0.9, r, len, 24), shell);
        bone.position.y = len / 2;
        pivot.add(bone);
        if (s === lengths.length - 1) {
          var tip = mesh(new THREE.SphereGeometry(r * 0.9, 24, 16), shell);
          tip.position.y = len;
          pivot.add(tip);
          var fingertip = mesh(new THREE.SphereGeometry(r * 0.55, 16, 12), accent);
          fingertip.position.set(0, len - 0.1, r * 0.55);
          pivot.add(fingertip);
        }
        pivots.push(pivot);
        p = pivot;
        r *= 0.88;
      }
      return pivots;
    }

    var joints = {};
    Object.keys(FINGERS).forEach(function (name) {
      var f = FINGERS[name];
      var base = new THREE.Group();
      base.position.set(f.x, f.y, 0);
      hand.add(base);
      var pivots = buildDigit(f.r, f.len, base);
      joints[name] = { spread: pivots[0], chain: pivots };
    });

    // Thumb: CMC base on the index side of the palm, then three segments
    var thumbBase = new THREE.Group();
    thumbBase.position.set(3.9, 2.4, 0.5);
    hand.add(thumbBase);
    var thumbOpp = new THREE.Group();
    thumbBase.add(thumbOpp);
    var thumbPivots = buildDigit(0.95, [3.0, 2.6, 2.2], thumbOpp);
    joints.thumb = { base: thumbBase, opp: thumbOpp, chain: thumbPivots };

    // Soft contact shadow
    var ground = new THREE.Mesh(new THREE.PlaneGeometry(60, 60), new THREE.ShadowMaterial({ opacity: 0.16 }));
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -4.7;
    ground.receiveShadow = true;
    scene.add(ground);

    // ------------------------------------------------------------ pose state
    var DIGITS = ['index', 'middle', 'ring', 'little', 'thumb'];
    var current = {}, goal = {};
    DIGITS.forEach(function (d) {
      current[d] = POSES.hold[d].slice();
      goal[d] = POSES.hold[d].slice();
    });
    var glow = new THREE.Color(0x8397a0), glowGoal = new THREE.Color(0x8397a0), glowLevel = 0, glowGoalLevel = 0;

    function apply() {
      ['index', 'middle', 'ring', 'little'].forEach(function (d) {
        var a = current[d], j = joints[d];
        j.chain[0].rotation.z = a[0];
        j.chain[0].rotation.x = a[1];
        j.chain[1].rotation.x = a[2];
        j.chain[2].rotation.x = a[3];
      });
      var t = current.thumb, jt = joints.thumb;
      jt.base.rotation.z = t[0];
      jt.opp.rotation.y = -t[1];
      jt.opp.rotation.x = t[1] * 0.55;
      jt.chain[1].rotation.x = t[2];
      jt.chain[2].rotation.x = t[3];
      accent.color.copy(glow);
      accent.emissive.copy(glow).multiplyScalar(glowLevel * 0.55);
    }

    // ------------------------------------------------------------ interaction
    var yaw = -0.38, pitch = 0.06, yawVel = 0, dragging = false, lastX = 0, lastY = 0, idle = 0;
    var el = renderer.domElement;
    el.style.touchAction = 'pan-y';
    el.addEventListener('pointerdown', function (e) {
      dragging = true; lastX = e.clientX; lastY = e.clientY; idle = 0;
      el.setPointerCapture(e.pointerId);
      container.classList.add('is-dragging');
    });
    el.addEventListener('pointermove', function (e) {
      if (!dragging) return;
      var dx = e.clientX - lastX, dy = e.clientY - lastY;
      lastX = e.clientX; lastY = e.clientY;
      yawVel = dx * 0.012;
      yaw += yawVel;
      pitch = Math.max(-0.5, Math.min(0.6, pitch + dy * 0.006));
    });
    function release() { dragging = false; container.classList.remove('is-dragging'); }
    el.addEventListener('pointerup', release);
    el.addEventListener('pointercancel', release);

    // ------------------------------------------------------------ render loop
    function resize() {
      var wpx = container.clientWidth, hpx = container.clientHeight;
      if (!wpx || !hpx) return;
      renderer.setSize(wpx, hpx, false);
      el.style.width = wpx + 'px';
      el.style.height = hpx + 'px';
      camera.aspect = wpx / hpx;
      camera.updateProjectionMatrix();
    }
    if (root.ResizeObserver) new ResizeObserver(resize).observe(container);
    resize();

    var last = performance.now(), visible = true;
    if (root.IntersectionObserver) {
      new IntersectionObserver(function (entries) { visible = entries[0].isIntersecting; }).observe(container);
    }
    function frame(now) {
      var dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      if (visible) {
        var k = reduceMotion ? 1 : 1 - Math.exp(-dt * 7);
        DIGITS.forEach(function (d) {
          for (var i = 0; i < 4; i++) current[d][i] += (goal[d][i] - current[d][i]) * k;
        });
        glow.lerp(glowGoal, k);
        glowLevel += (glowGoalLevel - glowLevel) * k;
        if (!dragging) {
          idle += dt;
          yawVel *= Math.pow(0.04, dt);
          yaw += yawVel;
          if (!reduceMotion && Math.abs(yawVel) < 0.002) yaw += Math.sin(idle * 0.5) * 0.0015;
        }
        hand.rotation.y = yaw;
        hand.rotation.x = pitch;
        apply();
        renderer.render(scene, camera);
      }
      root.requestAnimationFrame(frame);
    }
    apply();
    root.requestAnimationFrame(frame);

    return {
      setPose: function (name, color) {
        var pose = POSES[name] || POSES.hold;
        DIGITS.forEach(function (d) { goal[d] = pose[d].slice(); });
        glowGoal.set(color || 0x8397a0);
        glowGoalLevel = color ? 1 : 0;
      },
      resize: resize,
      _joints: joints
    };
  }

  root.Hand3D = { create: create, POSES: POSES };
})(this);
