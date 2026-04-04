## Section 1 — Brief Introduction

- Driver drowsiness is a persistent road-safety risk: reaction times lengthen, lane-keeping degrades, and microsleeps can produce sudden loss of control—outcomes that matter acutely in high-density Indian traffic with mixed road users.  
- National statistics underscore scale: **MoRTH’s *Road Accidents in India–2022*** (released via PIB) reported **4,61,312 road accidents**, **1,68,491 fatalities**, and **4,43,366 injuries** in a single year—figures also reflected in **NCRB’s Accidental Deaths & Suicides in India (ADSI) 2022** tabulations used for cross-verification of accidental death reporting.  
- **MoRTH’s annual *Road Accidents in India*** series remains the authoritative national baseline for trends, crash typing, and policy monitoring; it frames why software interventions must align with documented crash burdens rather than anecdote.  
- Indian tertiary neurotrauma and mental-health evidence further links severe road trauma to long-term burden: **NIMHANS-led work** (reported in national press) highlights substantial **RTA-related head injury caseloads** and **psychological sequelae among severely injured survivors**, reinforcing that prevention upstream (alertness monitoring) complements downstream care.  
- Complementary Indian epidemiology also associates **sleep disturbances with RTA risk** in population-representative analyses (e.g., **LASI-based modelling in *BMC Sleep***), supporting fatigue/sleepiness as a serious—often under-measured—determinant alongside speeding and rule violations commonly cited in official crash narratives.  
- Yet many “safety upgrades” remain **hardware-heavy** (dashcams with cloud analytics, fleet telematics, expensive biometric seats), creating a **gap for affordable, laptop/webcam-grade systems** deployable without subscriptions or fleet infrastructure.  
- **Computer vision** offers a contrasting path: **real-time facial geometry** can approximate eye openness using lightweight rules, enabling **continuous monitoring** on ordinary hardware—provided the method is **offline**, explainable, and robust enough for student/prototype deployment.  
- **VIGIL** is proposed as that **software-only, real-time, geometry-driven** countermeasure—closing the accessibility gap while targeting the same prevention objective as costlier instrumentation.

---

## Section 2 — Problem Statement

The problem objective is to **detect drowsiness in real time** by **processing live webcam frames with MediaPipe FaceMesh landmarks and Eye Aspect Ratio (EAR) geometry computed via NumPy/SciPy** in order to **reduce fatigue-related driving errors and attention failures that can culminate in collisions, near-miss events, or unsafe machine-operation lapses**. This work prioritizes a **transparent, lightweight pipeline** that flags sustained eye closure patterns rather than relying on cloud inference or bespoke datasets.

---

## Section 3 — SDG Relevance

**SDG 3 — Good Health and Well-Being (Target 3.6: halve deaths and injuries from road traffic crashes)**  
The project targets **preventable harm** by surfacing **impaired alertness before a critical failure**. By aiming to **interrupt drowsiness episodes early**, it supports safer mobility outcomes consistent with national road-injury reduction priorities documented in **MoRTH** and **NCRB** reporting.

**SDG 9 — Industry, Innovation and Infrastructure (Target 9.b: domestic technology development; broader access)**  
VIGIL demonstrates **domestic, low-footprint innovation**: a **CPU-friendly CV stack** that can run on common laptops with a webcam. This aligns with **widening access** to safety technologies beyond capital-intensive hardware ecosystems.

**SDG 11 — Sustainable Cities and Communities (Target 11.2: safe, sustainable transport systems)**  
Urban and intercity mobility in India mixes **vulnerable road users** with **high-variance driver behaviour**. A **real-time alert layer** contributes to **safer transport system operation** by mitigating a known human-factor failure mode—**drowsiness**—within everyday commuting and occupational driving contexts.

---

## Section 4 — Objectives

1. **Design** a modular pipeline that ingests **live video**, extracts **468-point FaceMesh landmarks**, and isolates **six periocular points per eye** for geometric analysis.  
2. **Develop** **EAR computation** using **Euclidean distances** (`scipy.spatial.distance.euclidean`) and combine left/right eyes into a **stable per-frame metric**.  
3. **Implement** **temporal thresholding** (EAR below **0.25** for **20 consecutive frames**) to distinguish **brief blinks** from **sustained closure**, triggering **audio + OpenCV overlays**.  
4. **Evaluate** system behaviour under **real-time constraints** (latency, tracking loss, lighting variation) using **on-screen HUD** feedback and **timestamped terminal logs** for alert events.  
5. **Demonstrate** a **fully offline** deployment with **no external dataset dependency**, suitable for **academic demonstration** and **repeatable benchmarking** on standard hardware.

---

## Section 5 — System Architecture / Block Diagram (Text Pipeline)

1. **Webcam Input (sensor)** — **Role:** supplies continuous RGB video to the application.  
2. **Frame Capture (`cv2.VideoCapture`)** — **Role:** reads frames, manages device index, and provides resolution/time sequencing for real-time processing.  
3. **Face Detection / Dense Mesh (`mediapipe.solutions.face_mesh.FaceMesh`)** — **Role:** estimates **468 3D landmarks** per face with tracking tuned for webcam streams.  
4. **Landmark Extraction (`FaceMesh.process` + landmark indexing)** — **Role:** converts normalized landmarks to **pixel coordinates** for geometric measurement on the current frame.  
5. **Eye Region Isolation (fixed periocular index lists)** — **Role:** selects **six points per eye** that bracket eyelid vertical separation and horizontal eye width.  
6. **EAR Computation (`numpy` arrays + `scipy.spatial.distance.euclidean`)** — **Role:** computes the **EAR ratio** from vertical and horizontal chord lengths for each eye and aggregates signal (e.g., average).  
7. **Threshold Evaluation (application logic + `config` constants)** — **Role:** increments a **frame counter** when **EAR < 0.25**; resets when eyes reopen; commits to **DROWSY** only after **20 consecutive** low-EAR frames.  
8. **State Routing** — **Role:** maintains **NORMAL** vs **DROWSY**; on transition to sustained drowsiness, triggers alerts.  
9. **Alert Trigger**  
   - **Visual (`cv2.rectangle`, `cv2.circle`, `cv2.putText`, `cv2.addWeighted`)** — **Role:** dark-theme HUD, eye boxes, landmark dots, semi-transparent **alert banner**.  
   - **Audio (`pygame.mixer.music` preferred; `playsound` fallback)** — **Role:** plays `assets/alert.wav` when drowsiness is confirmed; stops/ resets when the eye-open condition returns.

---

## Section 6 — Techniques and Dataset

**Techniques**

- **Facial landmark detection** — Uses **MediaPipe FaceMesh** to localize stable facial structure for per-frame measurement without training project-specific detectors.  
- **Eye Aspect Ratio (EAR)** — Computes a **normalized openness index** from **six eye points**, sensitive to eyelid separation changes during closure.  
- **Temporal frame analysis** — Applies a **consecutive-frame counter** so transient blinks do not equal drowsiness.  
- **Real-time video processing** — Maintains a **continuous OpenCV loop** (`read` → `imshow` → `waitKey`) suitable for interactive deployment.  
- **Geometric distance computation** — Implements EAR via **Euclidean distances** between selected landmark pairs, emphasizing interpretability over black-box scoring.

**Dataset**

- **No external dataset was used.**  
- The system relies on **geometric rules**, not dataset-learned class boundaries: **EAR is a derived ratio** and is **broadly comparable across faces** when landmarks track correctly. **MediaPipe FaceMesh** supplies a **pretrained mesh estimator** internally; the project does not train or fine-tune that model.  
- **Advantage:** **zero custom data dependency**, **full offline operation**, and **transparent failure modes** (tracking loss, occlusion) that can be diagnosed from overlays rather than opaque model scores.

---

## Section 7 — Methodology (Elaborated)

**Phase 1 — Video Acquisition**  
Frames are acquired through **`cv2.VideoCapture`** using a configured camera index. Each iteration pulls a **BGR frame**; the pipeline assumes stable capture for timing, while resolution follows the webcam’s default unless explicitly set. This phase defines the **real-time clock** against which temporal drowsiness logic operates.

**Phase 2 — Face and Landmark Detection**  
The frame is converted to **RGB** for **`FaceMesh.process`**, which returns **468 landmarks** including refined regions when enabled. **Detection/tracking confidence thresholds** reduce spurious jumps under motion. Landmarks are accessed via **`multi_face_landmarks[0].landmark[i]`** for indexed coordinates.

**Phase 3 — Eye Region Extraction**  
For each eye, **six periocular indices** are selected to represent **outer corner, upper lid pairs, inner corner, and lower lid pairs**. Pixel mapping uses normalized \(x,y\) scaled by frame width/height, yielding stable point sets for distance measurement.

**Phase 4 — EAR Computation**  
The implemented ratio is  
\[
\mathrm{EAR}=\frac{\lVert p_2-p_6\rVert+\lVert p_3-p_5\rVert}{2\times \lVert p_1-p_4\rVert}
\]  
where **\(p_1\)–\(p_4\)** span the **horizontal eye aperture**, and **\((p_2,p_6)\)** and **\((p_3,p_5)\)** capture **vertical separations** across the eyelids. **Eye closure** shrinks vertical distances relative to horizontal extent, **dropping EAR sharply** compared with the open-eye configuration.

**Phase 5 — Temporal Thresholding**  
An **EAR threshold of 0.25** defines “closed-enough.” A **counter increments only while** the condition holds across frames; **blinks** typically **recover quickly**, resetting the counter, whereas **drowsy closure** sustains low EAR long enough to reach **20 consecutive frames**, confirming a **drowsy state** under the project’s conservative rule.

**Phase 6 — Alert Mechanism**  
On confirmed drowsiness, **visual alerting** uses **OpenCV drawing primitives** and **alpha blending** for a **semi-transparent red banner**; **audio** uses **`pygame.mixer.music.play`** (looping) when available. **Reset** occurs when **EAR returns above threshold**, clearing the counter and stopping audio (`pygame.mixer.music.stop`), restoring **NORMAL** operation.

---

## Section 8 — Computer Vision Implementation (Brief)

- **Frame loop:** `cv2.VideoCapture.read()` pulls each frame; the main loop drives **continuous inference** and display via `cv2.imshow()`.  
- **Color conversion:** `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` prepares input for **MediaPipe**, which expects **RGB** ordering.  
- **Mesh inference:** `mediapipe.solutions.face_mesh.FaceMesh(...)` constructs the solver; `FaceMesh.process(rgb_frame)` returns **`multi_face_landmarks`** when a face is tracked.  
- **Landmark access:** landmark coordinates are read from **`face_landmarks.landmark[index]`**, converting normalized coordinates to pixels for **NumPy** arrays (`dtype=float32`).  
- **Distance math:** `scipy.spatial.distance.euclidean` evaluates segment lengths used in **EAR**; left/right EAR values are combined (e.g., averaged) for a single control signal.  
- **Geometry overlays:** `cv2.circle` renders **lime landmark dots**; `cv2.boundingRect` supports **`cv2.rectangle`** eye boxes in **electric blue**.  
- **HUD text:** `cv2.putText` (with `cv2.FONT_HERSHEY_SIMPLEX`) prints **cyan EAR** and counters; centered status text uses `cv2.getTextSize` for alignment.  
- **Alert blending:** `cv2.addWeighted` composites a tinted banner region for **semi-transparent red** alert emphasis on the live frame.  
- **Recording (optional):** `cv2.VideoWriter` (when enabled in `config.py`) writes annotated frames to disk for audit and demonstration.