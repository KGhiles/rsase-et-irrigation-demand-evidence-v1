# Supplementary Methods S1. Known-cold reference sensitivity re-solve

**Terminology.** Visible text uses ETrF. Historical code and source-table fields may retain `ETf`.

## S1.1. Purpose and analytical boundary

The known-cold reference re-solve quantified the sensitivity of accepted 2011 METRIC output to the cold side of the image-scale calibration. The accepted target-scene hot reference, surface temperature, net radiation, soil heat flux, meteorological forcing, and canopy inputs were retained. A same-date, actively transpiring source statistic near Etter, Texas replaced the accepted cold-reference temperature. The full sensible-heat and stability pathway was then re-solved.

The experiment generated diagnostic alternative ETrF and ETa-equivalent fields. It did not replace the accepted METRIC product, validate ET, estimate pumping, infer groundwater withdrawal, or identify water source. The strict summaries describe reference sensitivity within the same-date source-supported subset.

## S1.2. Same-date source-reference definition

The source locator was 35.999399 degrees N, 101.979024 degrees W. Circular source windows of 1000 and 3000 m were evaluated separately. The locator defined a search center only; it did not represent a field boundary, pivot, exact anchor pixel, validation site, well, or management unit.

Source pixels came from the accepted Moore County scene acquired on the same date as the target scene. The main analysis did not use temporal fallback. Each source pixel satisfied the accepted-valid mask, the CDL corn mask, and the active-canopy mask. Additional source thresholds were NDVI >= 0.65, LAI >= 2.50, LE/Rn >= 0.85, accepted ETrF >= 0.85, and H <= 50 W m-2.

The source temperature for radius r was the median surface temperature of the eligible source pixels. A source-radius pair was valid with at least 30 pixels. The 25th-percentile temperature was retained as an audit statistic but was not used in the primary strict re-solve.

$$ C_r = {p in B_r: M_acc = 1, M_corn = 1, NDVI >= 0.65, LAI >= 2.50, LE/Rn >= 0.85, ETrF >= 0.85, H <= 50 W m^-2} 	ag{S1} $$
*Eligible source-pixel set.*

$$ T_KC,r = median{T_s,p : p in C_r},  n(C_r) >= 30 	ag{S2} $$
*Primary known-cold temperature.*

$$ Delta T_anchor,r = T_hot - T_KC,r >= 6.5 K 	ag{S3} $$
*Main thermal-separation gate.*

## S1.3. Frame eligibility and retained target inputs

Source and target records were joined by scene date and source radius. The accepted target scene supplied the hot-reference condition and all target-scene inputs. The main thermal-separation gate required the accepted hot-reference temperature to exceed the imposed known-cold temperature by at least 6.5 K. Audit indicators at 1.0 and 4.0 K were exported but did not replace the 6.5 K gate.

A frame entered the re-solve only after source validity, minimum source-pixel support, same-date matching, required target inputs, and the 6.5 K thermal-separation gate were satisfied. Failed frames remained in the audit tables and were excluded from strict summaries.

## S1.4. Cold-reference energy-balance condition

For each accepted target scene and source radius, the imposed cold-reference temperature equaled the same-date source median. Source-pool medians supplied cold-reference net radiation, soil heat flux, LAI, air density, wind speed, and air temperature. The cold-reference latent-energy term used the target-scene hourly alfalfa-reference ET convention. The factor 1.05 was applied only to the imposed cold-reference latent-energy term.

The accepted hot-reference sensible heat was retained as Rn,hot - Ghot, corresponding to the dry-reference condition used by the accepted METRIC workflow. The hot-reference state was not replaced by the source frame.

$$ LE_cold,r^KC = 1.05 lambda ETr_h / 3600 	ag{S4} $$
*Cold-reference latent-energy condition.*

$$ H_cold,r^KC = R_n,cold,r - G_cold,r - LE_cold,r^KC 	ag{S5} $$
*Cold-reference sensible heat.*

$$ H_hot = R_n,hot - G_hot 	ag{S6} $$
*Retained hot-reference sensible heat.*

## S1.5. Roughness and neutral initialization

Canopy height and roughness terms were recalculated from the target-pixel LAI and from the source-pool median LAI. LAI was bounded to 0-7. Canopy height was LAI/3 and was bounded to 0.10-2.0 m. Displacement height was 0.67 times canopy height. Momentum roughness was 0.123 times canopy height with a 0.005 m lower bound. Heat roughness was one tenth of momentum roughness with a 0.0005 m lower bound.

The first aerodynamic state assumed neutral stability. Wind and temperature measurement heights were both 2 m. Effective height above displacement was bounded below by 0.10 m. Friction velocity was bounded below by 0.05 m s-1, aerodynamic resistance was bounded below by 1 s m-1, and the final iterative resistance was capped at 8000 s m-1. Cap hits were exported as QA flags.

$$ h_c = clip(LAI/3, 0.10, 2.0) 	ag{S7} $$
*Canopy height.*

$$ d = 0.67 h_c;  z_om = max(0.123 h_c, 0.005);  z_oh = max(z_om/10, 0.0005) 	ag{S8} $$
*Roughness and displacement.*

$$ u_* = max[0.05, kappa u_2 / {ln(z_eff/z_om) - psi_m}] 	ag{S9} $$
*Friction velocity.*

$$ r_ah,raw = max[1, {ln(z_u/z_oh)-psi_h,u}{ln(z_t/z_oh)-psi_h,t} / (kappa^2 u_*)];  r_ah = min(r_ah,raw, 8000) 	ag{S10} $$
*Aerodynamic resistance.*

## S1.6. Image-scale dT-Ts calibration

The near-surface temperature differences at the retained hot reference and imposed known-cold reference were calculated from sensible heat, aerodynamic resistance, air density, and specific heat. The two reference states defined a linear dT-Ts relation. The relation was recalculated during each stability iteration because hot and cold aerodynamic resistance changed with stability.

$$ dT_cold = H_cold r_ah,cold / (rho_cold c_p);  dT_hot = H_hot r_ah,hot / (rho_hot c_p) 	ag{S11} $$
*Reference dT values.*

$$ a = (dT_hot - dT_cold)/(T_hot - T_KC,r);  b = dT_cold - a T_KC,r 	ag{S12} $$
*Image-scale calibration.*

$$ dT_i = a T_s,i + b;  H_i = rho_i c_p dT_i / r_ah,i 	ag{S13} $$
*Target-pixel sensible heat.*

## S1.7. Monin-Obukhov stability iteration

Monin-Obukhov length was computed from sensible heat, friction velocity, air density, air temperature, and the von Karman constant. A signed 5 W m-2 floor stabilized H near zero, and L was bounded to -1000 to 1000 m. Stable conditions used psi_m = psi_h = -5 zeta. Unstable conditions used the standard Businger-Dyer forms stated below.

Target-pixel and cold-reference aerodynamic states were updated together. Each iteration recalculated stability corrections, friction velocity, aerodynamic resistance, hot and cold dT values, the dT-Ts calibration, and target-pixel sensible heat. The solver executed at most 10 iterations. A pixel converged when the absolute H change was no greater than 2 W m-2 and the fractional aerodynamic-resistance change was no greater than 0.02. The first satisfying iteration was retained as the convergence count; pixels failing the final criterion were excluded from the valid-output mask.

$$ H_safe = sign(H) max(|H|, 5 W m^-2) 	ag{S14} $$
*Numerical H floor.*

$$ L = -rho c_p u_*^3 T_a / (kappa g H_safe),  clipped to [-1000, 1000] m 	ag{S15} $$
*Monin-Obukhov length.*

$$ x = (1 - 16 zeta)^(1/4),  zeta < 0 	ag{S16} $$
*Unstable stability variable.*

$$ psi_m = 2 ln[(1+x)/2] + ln[(1+x^2)/2] - 2 atan(x) + pi/2 	ag{S17} $$
*Unstable momentum correction.*

$$ psi_h = 2 ln[(1+x^2)/2] 	ag{S18} $$
*Unstable heat correction.*

$$ psi_m = psi_h = -5 zeta,  zeta >= 0 	ag{S19} $$
*Stable correction.*

$$ |H^(k) - H^(k-1)| <= 2 W m^-2  and  |r_ah^(k)-r_ah^(k-1)|/max(|r_ah^(k-1)|,1) <= 0.02 	ag{S20} $$
*Convergence criterion.*

## S1.8. Recomputed latent heat, ETrF, and ETa-equivalent response

After the final stability iteration, latent heat was computed as the residual of net radiation, soil heat flux, and re-solved sensible heat. The final reference latent-energy denominator used target-scene hourly ETr without the 1.05 cold-reference multiplier. A lower numerical denominator of 1 W m-2 prevented division by zero.

Raw re-solved ETrF was flagged at the lower and upper clamp limits and then bounded to -0.1 to 1.2. Daily ETa-equivalent depth was the bounded re-solved ETrF multiplied by target-scene daily ETr. Perturbation fields were calculated relative to the accepted baseline ETrF and ETa-equivalent depth.

$$ LE_i^KC = R_n,i - G_i - H_i^KC 	ag{S21} $$
*Re-solved latent heat.*

$$ LE_ref = lambda ETr_h / 3600;  ETrF_raw,i^KC = LE_i^KC / max(LE_ref, 1 W m^-2) 	ag{S22} $$
*Raw re-solved ETrF.*

$$ ETrF_i^KC = clip(ETrF_raw,i^KC, -0.1, 1.2);  ETa_i^KC = ETrF_i^KC ETr_daily 	ag{S23} $$
*Bounded ETrF and daily ETa-equivalent.*

$$ Delta ETrF_i^KC = ETrF_i^KC - ETrF_base,i;  Delta ETa_i^KC = ETa_i^KC - ETa_base,i 	ag{S24} $$
*Perturbation response.*

## S1.9. Physical validity and strict mask hierarchy

The physical-output mask required Rn > 20 W m-2, r_ah > 0, -300 < LE < 1000 W m-2, -500 < H < 800 W m-2, and -1 < raw ETrF < 2. The valid-output mask also required the accepted baseline mask, a valid same-date source, the 6.5 K thermal gap, and numerical convergence.

ACES summaries imposed a nested interpretation hierarchy. The active-canopy strict mask was the intersection of valid output and the active-canopy mask. The large-ETrF strict mask added the baseline large-ETrF active-canopy mask. The mechanism strict mask further required the physical-coherence mechanism mask or its exported overlap mask. This construction guaranteed valid output >= active canopy >= large ETrF >= mechanism evidence.

$$ M_valid \supseteq M_active \supseteq M_large-ETrF \supseteq M_mechanism 	ag{S25} $$
*Strict nested hierarchy from broad to narrow.*

## S1.10. Sensitivity summaries and interpretation

Differences were summarized by county, source radius, raster, and strict mask domain. Primary summaries used the median absolute ETa-equivalent response. A response of at least 1.0 mm d-1 in absolute value was classified as material sensitivity.

The final strict set contained 39 readable, valid, hierarchy-passing rasters: 30 from Moore and 9 from Ochiltree. Dallam and Hartley lacked retained same-date source-supported rasters and were excluded from the strict comparison. Their absence did not invalidate the accepted baseline scenes. The retained county-radius groups had zero nonconvergent fractions and zero aerodynamic-resistance cap-hit fractions. Clamp fractions remained low and were reported as numerical QA.

The re-solved fields were used only to quantify reference dependence. The accepted METRIC fields remained the production outputs.

$$ |Delta ETa_i^KC| >= 1.0 mm d^-1 	ag{S26} $$
*Material-sensitivity threshold.*

## Table S4. Fixed parameters used in the strict known-cold re-solve

| Parameter | Value | Role |
|---|---:|---|
| Source locator | 35.999399 N, 101.979024 W | Search center only |
| Source radii | 1000 and 3000 m | Spatial-support sensitivity |
| Temporal matching | Same date only | Main strict analysis |
| Minimum source pixels | 30 | Source validity |
| Primary source statistic | Median | Known-cold temperature and source state |
| Thermal-separation gate | 6.5 K | Main preflight gate |
| Maximum stability iterations | 10 | Numerical solver |
| H convergence tolerance | 2 W m-2 | Pixel convergence |
| r_ah convergence tolerance | 0.02 fractional change | Pixel convergence |
| r_ah cap | 8000 s m-1 | Numerical guardrail |
| ETrF clamp | -0.1 to 1.2 | Output guardrail after flagging |
| Material |Delta ETa| | 1.0 mm d-1 | Interpretation threshold |

## Interpretation boundary

The known-cold fields are diagnostic sensitivity outputs. They are not corrected ET, validation ET, observed ET, pumping estimates, groundwater-withdrawal estimates, or a replacement METRIC product.
