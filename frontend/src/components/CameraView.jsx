import React, { useRef, useEffect } from 'react';
import './CameraView.css';

// Dessine les lignes de mesure (EAR/MAR) sur le canvas overlay
// points : tableau de [x,y] normalisés (0-1) — même ordre que draw_metric_lines backend
// color  : chaîne CSS ex. '#00FFFF'
function drawMetricLines(ctx, points, color, canvasW, canvasH) {
  if (!points || points.length < 6) return;

  const toPixel = ([x, y]) => [x * canvasW, y * canvasH];

  const [p0, p1, p2, p3, p4, p5] = points.map(toPixel);

  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.beginPath();

  // Vertical 1 : p1 — p5
  ctx.moveTo(p1[0], p1[1]);
  ctx.lineTo(p5[0], p5[1]);

  // Vertical 2 : p2 — p4
  ctx.moveTo(p2[0], p2[1]);
  ctx.lineTo(p4[0], p4[1]);

  // Horizontal  : p0 — p3
  ctx.moveTo(p0[0], p0[1]);
  ctx.lineTo(p3[0], p3[1]);

  ctx.stroke();
}

const CameraView = ({
  videoRef,
  canvasRef,          // canvas caché pour capture
  annotatedImage,     // renommé : contient maintenant les annotations {left_eye, right_eye, mouth}
  emotion,
  error,
  vlmQuestion,
  onVLMResponse,
  temperature,
  primaryEmotion,
}) => {
  const overlayCanvasRef = useRef(null);

  const emotionLabel = primaryEmotion
    ? primaryEmotion.charAt(0).toUpperCase() + primaryEmotion.slice(1)
    : '';

  // Dessine les annotations sur le canvas overlay à chaque mise à jour
  useEffect(() => {
    const canvas = overlayCanvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    // Synchroniser la taille du canvas avec l'élément vidéo affiché
    const w = video.clientWidth || video.videoWidth || 640;
    const h = video.clientHeight || video.videoHeight || 480;
    canvas.width = w;
    canvas.height = h;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, w, h);

    if (!annotatedImage) return; // pas d'annotations reçues

    const { left_eye, right_eye, mouth } = annotatedImage;

    drawMetricLines(ctx, left_eye, '#00FFFF', w, h); // cyan  – oeil gauche
    drawMetricLines(ctx, right_eye, '#00FFFF', w, h); // cyan  – oeil droit
    drawMetricLines(ctx, mouth, '#FF6400', w, h); // orange – bouche
  }, [annotatedImage, videoRef]);

  console.log('CameraView primaryEmotion:', { primaryEmotion, emotionLabel });

  return (
    <div className="camera-container">
      {error ? (
        <div className="error-message">
          <div>{error}</div>
          <button className="retry-camera" onClick={() => {
            if (typeof window.__RETRY_CAMERA__ === 'function') window.__RETRY_CAMERA__();
          }}>Réessayer la caméra</button>
        </div>
      ) : (
        <>
          <div className="video-wrapper">
            {/* Flux vidéo natif – toujours en temps réel */}
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="video-feed"
            />

            {/* Canvas transparent superposé – annotations dessinées côté client */}
            <canvas
              ref={overlayCanvasRef}
              className="annotation-overlay"
            />

            {/* Indicateur de confort/inconfort */}
            {primaryEmotion && (
              <div className="comfort-indicator">
                {emotionLabel}
              </div>
            )}

            {/* Jauge de température */}
            {typeof temperature === 'number' && (
              <div className="temperature-indicator">
                <div className="temp-gauge-mini">
                  <div
                    className="temp-gauge-mini-fill"
                    style={{ height: `${(temperature / 50) * 100}%` }}
                  />
                </div>
                <span className="temp-value-mini">{temperature.toFixed(1)}°C</span>
              </div>
            )}

            {/* Question VLM */}
            {vlmQuestion && (
              <div className="vlm-question-overlay">
                <div className="vlm-question-box">
                  <p className="vlm-question-text">{vlmQuestion}</p>
                  <div className="vlm-action-buttons">
                    <button
                      className="vlm-button vlm-button-yes"
                      onClick={() => onVLMResponse('oui')}
                    >
                      OUI
                    </button>
                    <button
                      className="vlm-button vlm-button-no"
                      onClick={() => onVLMResponse('non')}
                    >
                      NON
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Canvas caché pour capture de frame */}
          <canvas ref={canvasRef} style={{ display: 'none' }} />
        </>
      )}
    </div>
  );
};

export default CameraView;
