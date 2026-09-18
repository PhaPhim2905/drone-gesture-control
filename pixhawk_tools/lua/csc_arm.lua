-- csc_arm.lua: ARM / DISARM bằng CHỤM HAI CẦN VÀO TRONG (tay cầm Mode 2).
--
--   cần TRÁI  xuống góc dưới-PHẢI  (ga thấp nhất + yaw phải)
--   cần PHẢI  xuống góc dưới-TRÁI  (pitch kéo về + roll trái)
--
--   ARM     chụm giữ 1.5 s -> "buong can de ARM" -> BUÔNG cả 2 cần trong 3 s
--   DISARM  đang arm và ĐÃ Ở MẶT ĐẤT: chụm giữ 1.5 s -> tắt ngay
--
-- Vì sao phải buông mới arm: ArduPilot từ chối arm khi roll/pitch/yaw chưa về giữa
-- ("Arm: Roll (RC1) is not neutral", đo trên SITL 15/9). Kiểm tra đó giữ nguyên.
--
-- Tay cầm UniRC7 có cần GA LÒ XO (buông tay về giữa ~1495). Nên lúc buông chỉ đòi ga
-- KHÔNG CAO HƠN giữa. ArduPilot cho arm với ga ở giữa trong LOITER/ALT_HOLD
-- ("Throttle too high" chỉ khi đòi leo, AP_Arming_Copter.cpp); STABILIZE/ACRO vẫn đòi
-- ga 0 -> với ga lò xo KHÔNG dùng hai mode đó. Pixhawk đặt PILOT_THR_BHV = 1.
--
-- Cài:
--   SITL        ./sim/run_sitl.sh ... --proposal   (tự chép vào <SITL_DIR>/scripts/)
--   Pixhawk 6X  chép file này vào thẻ nhớ APM/scripts/, đặt SCR_ENABLE=1, khởi động lại,
--               ARMING_RUDDER=0 để chỉ còn một cách arm bằng cần.
--
-- An toàn:
--   - ARM đi qua kiểm tra PreArm như mọi cách arm khác (arming:arm()).
--   - DISARM chỉ khi vehicle:get_likely_flying() = false. Đang bay chụm cần KHÔNG tắt động cơ;
--     tắt khẩn cấp là việc của công tắc E-stop (RC6_OPTION = 31).
--   - Mất sóng tay cầm: bỏ qua.
--   - Sau mỗi lần kích hoạt phải buông cần về giữa mới kích hoạt lại được.
--
-- Chiều cần theo quy ước ArduPilot sau Radio Calibration: kênh 1 thấp = roll trái,
-- kênh 2 cao = pitch kéo về, kênh 3 thấp = ga thấp, kênh 4 cao = yaw phải.
-- ĐÃ THỬ TRÊN SITL. Trên tay cầm thật: thử qua sim/rc_bridge.py trước khi nạp vào 6X.

local HOLD_MS  = 1500
local EDGE     = 0.85     -- |cần| >= 0.85 coi là đã chạm góc
local CENTER   = 0.5      -- mọi cần về dưới mức này = đã buông
local PERIOD   = 50
local MSG_INFO, MSG_WARN = 6, 4

local function ch(name)
  return rc:get_channel(math.floor(param:get(name)))
end

local c_roll, c_pitch = ch("RCMAP_ROLL"), ch("RCMAP_PITCH")
local c_thr, c_yaw    = ch("RCMAP_THROTTLE"), ch("RCMAP_YAW")

local ARM_WINDOW_MS = 3000
local THR_MID_MAX   = 0.10   -- ga lò xo về giữa = 0; cao hơn chút coi như đang đẩy ga

local t_start = nil
local need_release = false
local arm_pending_until = nil   -- đã chụm đủ, chờ buông cần để arm

local function update()
  if not (c_roll and c_pitch and c_thr and c_yaw) then
    gcs:send_text(MSG_WARN, "CSC: khong doc duoc RCMAP_*, script dung")
    return
  end
  if not rc:has_valid_input() then
    t_start, arm_pending_until = nil, nil
    return update, PERIOD
  end

  local thr   = c_thr:norm_input_ignore_trim()   -- -1 = ga thấp nhất
  local yaw   = c_yaw:norm_input()               -- +1 = yaw phải
  local pitch = c_pitch:norm_input()             -- +1 = kéo về
  local roll  = c_roll:norm_input()              -- -1 = roll trái

  local inward = thr <= -EDGE and yaw >= EDGE and pitch >= EDGE and roll <= -EDGE
  -- "về giữa" theo đúng vùng chết RCn_DZ mà kiểm tra arm của ArduPilot dùng
  local neutral = c_roll:norm_input_dz() == 0 and c_pitch:norm_input_dz() == 0
                  and c_yaw:norm_input_dz() == 0

  if arm_pending_until then
    local now = millis()
    if neutral and thr <= THR_MID_MAX then
      arm_pending_until = nil
      need_release = false
      if arming:is_armed() then
        return update, PERIOD
      end
      if arming:arm() then
        gcs:send_text(MSG_INFO, "CSC: ARM")
      else
        gcs:send_text(MSG_WARN, "CSC: ARM bi tu choi - xem PreArm")
      end
    elseif now > arm_pending_until then
      arm_pending_until = nil
      gcs:send_text(MSG_WARN, "CSC: qua 3 s khong buong can (hoac ga cao hon giua) - huy ARM")
    end
    return update, PERIOD
  end

  if need_release then
    if math.abs(yaw) < CENTER and math.abs(pitch) < CENTER and math.abs(roll) < CENTER then
      need_release = false
    end
    return update, PERIOD
  end

  if not inward then
    t_start = nil
    return update, PERIOD
  end

  local now = millis()
  if t_start == nil then
    t_start = now
    return update, PERIOD
  end
  if (now - t_start) < HOLD_MS then
    return update, PERIOD
  end

  t_start = nil
  need_release = true
  if not arming:is_armed() then
    arm_pending_until = now + ARM_WINDOW_MS
    gcs:send_text(MSG_INFO, "CSC: buong can de ARM")
  elseif vehicle:get_likely_flying() then
    gcs:send_text(MSG_WARN, "CSC: dang bay - KHONG disarm. Tat khan cap = E-stop kenh 6")
  else
    arming:disarm()
    gcs:send_text(MSG_INFO, "CSC: DISARM")
  end
  return update, PERIOD
end

gcs:send_text(MSG_INFO, "CSC: chum 2 can vao trong 1.5 s de ARM/DISARM")
return update, 1000
