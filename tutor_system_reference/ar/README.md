# AR Mode (Design + Implementation Notes)

This repo includes a placeholder for AR integration. The recommended approach is:

## 1) UX rules (as specified)
- Opt-in and contextual: AR appears as "Visual Mode" suggestion, not default.
- Fail gracefully: if camera permission fails, automatically use Visual 2D Mode.
- Fast: AR should open within <10 seconds after button press.
- Pedagogically aligned: AR targets specific misconceptions (e.g., balance model for 'moving terms').

## 2) Minimal v1 module: Balance Scale Algebra
Represent an equation as a balance scale. Operations must apply to both sides.
This directly addresses the misconception: "moving terms across" without inverse operations.

## 3) Practical implementation options
A) WebAR (fast iteration): 8th Wall / WebXR (device support varies)
B) Mobile (robust): Unity + AR Foundation (iOS/Android)
C) Hybrid: React Native wrapper + native AR module

## 4) Integration contract
The tutor UI requests AR suggestion via policy response:
- action: SUGGEST_AR_VISUAL
- ui_suggestions.show_ar_button = true
- ui_suggestions.fallback_mode = VISUAL_2D

The UI must:
1) Attempt to open camera (AR)
2) If denied/unavailable, switch to 2D animation immediately (no blocking)
