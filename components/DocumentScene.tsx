"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";

export function DocumentScene() {
  const mount = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const element = mount.current;
    if (!element) return;

    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x071015, 0.09);

    const camera = new THREE.PerspectiveCamera(36, 1, 0.1, 100);
    camera.position.set(0, 0.2, 8.2);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x000000, 0);
    element.appendChild(renderer.domElement);

    const group = new THREE.Group();
    scene.add(group);

    const pageMaterial = new THREE.MeshStandardMaterial({
      color: 0xe8f4ef,
      roughness: 0.42,
      metalness: 0.12,
      emissive: 0x10251e,
      emissiveIntensity: 0.35,
    });
    const edgeMaterial = new THREE.MeshStandardMaterial({ color: 0x42d7bc, metalness: 0.7, roughness: 0.2 });
    const lineMaterial = new THREE.MeshBasicMaterial({ color: 0x5f8e88, transparent: true, opacity: 0.63 });

    const sheets: THREE.Group[] = [];
    [-0.72, 0, 0.72].forEach((offset, index) => {
      const sheet = new THREE.Group();
      const page = new THREE.Mesh(new THREE.BoxGeometry(3.2, 4.1, 0.075), pageMaterial);
      page.position.z = index * -0.13;
      sheet.add(page);

      const edge = new THREE.Mesh(new THREE.BoxGeometry(3.26, 4.16, 0.025), edgeMaterial);
      edge.position.z = index * -0.16 - 0.06;
      sheet.add(edge);

      for (let line = 0; line < 8; line += 1) {
        const width = line === 0 ? 1.45 : 2.25 - (line % 3) * 0.26;
        const textLine = new THREE.Mesh(new THREE.PlaneGeometry(width, 0.045), lineMaterial);
        textLine.position.set(-0.28, 1.25 - line * 0.31, 0.045 + index * -0.13);
        sheet.add(textLine);
      }

      sheet.position.set(offset * 0.23, -0.1, -index * 0.08);
      sheet.rotation.set(offset * 0.08, offset * 0.1, offset * -0.08);
      group.add(sheet);
      sheets.push(sheet);
    });

    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(2.85, 0.015, 8, 100),
      new THREE.MeshBasicMaterial({ color: 0x29b3de, transparent: true, opacity: 0.48 })
    );
    ring.rotation.x = Math.PI / 2.8;
    ring.position.z = -0.8;
    scene.add(ring);

    const points = new THREE.Points(
      new THREE.BufferGeometry().setFromPoints(
        Array.from({ length: 100 }, () => new THREE.Vector3(
          (Math.random() - 0.5) * 12,
          (Math.random() - 0.5) * 9,
          -Math.random() * 5
        ))
      ),
      new THREE.PointsMaterial({ color: 0x77e3c3, size: 0.025, transparent: true, opacity: 0.7 })
    );
    scene.add(points);

    const keyLight = new THREE.DirectionalLight(0xb6ffe7, 2.2);
    keyLight.position.set(4, 5, 4);
    scene.add(keyLight);
    const fillLight = new THREE.PointLight(0x27b7de, 12, 13);
    fillLight.position.set(-4, -1, 3);
    scene.add(fillLight);

    let frame = 0;
    const resize = () => {
      const { width, height } = element.getBoundingClientRect();
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
    };
    const animate = () => {
      frame = requestAnimationFrame(animate);
      const time = performance.now() * 0.00035;
      group.rotation.y = Math.sin(time) * 0.28 - 0.28;
      group.rotation.x = Math.cos(time * 1.3) * 0.06;
      sheets.forEach((sheet, index) => { sheet.position.y = -0.1 + Math.sin(time * 2 + index) * 0.08; });
      ring.rotation.z = time * 0.85;
      points.rotation.y = -time * 0.23;
      renderer.render(scene, camera);
    };

    resize();
    animate();
    window.addEventListener("resize", resize);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("resize", resize);
      renderer.dispose();
      element.removeChild(renderer.domElement);
    };
  }, []);

  return <div className="document-scene" ref={mount} aria-hidden="true" />;
}
