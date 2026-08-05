"""Boshning burilishini InsightFace bergan 5 ta kalit nuqtadan hisoblaydi.

# Nima uchun kerak

Passiv anti-spoofing (MiniFASNet) bosma suratni va ekrandagi tasvirni yaxshi
to'sadi, chunki ularda tekstura va yorug'lik boshqacha. Lekin u OLDINDAN
YOZILGAN videoni to'sa olmaydi: video haqiqiy odamning haqiqiy yuzidan olingan,
ya'ni ta'rifga ko'ra "tirik" ko'rinadi.

Aktiv tekshiruv shu bo'shliqni yopadi: server har safar TASODIFIY buyruq beradi
("chapga qarang"), va javob shu buyruqqa mos kelishi kerak. Yozib qo'yilgan
videoda buyruq oldindan ma'lum emas, shuning uchun mos javob bo'lishi mumkin
emas.

# Nima uchun faqat YAW ishonchli

InsightFace `detection` moduli 5 ta nuqta beradi:
    0 — chap ko'z, 1 — o'ng ko'z, 2 — burun, 3 — chap lab burchagi, 4 — o'ng lab

Yaw (gorizontal burilish) bu nuqtalardan yaxshi chiqadi: bosh burilganda burun
ko'zlar o'rtasidan sezilarli siljiydi va bu siljish ko'zlar orasidagi masofaga
nisbatan o'lchanadi — ya'ni kameragacha bo'lgan masofadan deyarli mustaqil.

Pitch (yuqoriga/pastga) va roll (yonboshlash) esa 5 nuqtadan ancha noaniq:
pitch uchun yuzning uch o'lchamli modeli kerak, roll esa ko'zlar chizig'idan
olinadi, lekin uni kamerani egish bilan ham hosil qilib bo'ladi. Shuning uchun
ular hisoblanadi, lekin AKTIV TEKSHIRUVDA ISHLATILMAYDI — qaror faqat yaw
bo'yicha chiqariladi.

# Cheklov — ochiq aytiladi

Bu real vaqtdagi deepfake'ni to'smaydi: buyruqqa javob bera oladigan tizim
har qanday buyruqni bajaradi. Aktiv tekshiruv faqat OLDINDAN YOZILGAN
materialga qarshi ishlaydi. Deepfake'ga qarshi passiv model va boshqa
qatlamlar javob beradi.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Kalit nuqtalar tartibi — InsightFace SCRFD standarti.
LEFT_EYE = 0
RIGHT_EYE = 1
NOSE = 2
LEFT_MOUTH = 3
RIGHT_MOUTH = 4

# Ko'zlar orasidagi masofa shu qiymatdan kichik bo'lsa hisob ishonchsiz:
# yuz juda kichik yoki nuqtalar noto'g'ri joylashgan.
MIN_EYE_DISTANCE_PX = 20.0

# Burun siljishini gradusga o'girish koeffitsiyenti.
#
# Qiymat empirik: normallashtirilgan siljish (burun ofseti / ko'zlar orasi)
# taxminan ±0.5 oralig'ida o'zgaradi va bu ~±45 gradusga to'g'ri keladi.
# Aniq antropometrik model emas — aktiv tekshiruv uchun kerak ham emas,
# chunki qaror "chapgami yoki o'nggami" degan savolga tayanadi, aniq
# gradusga emas.
_YAW_SCALE_DEG = 90.0


@dataclass(frozen=True)
class HeadPose:
    """Boshning burilishi (gradus)."""

    yaw: float
    pitch: float
    roll: float
    available: bool

    @classmethod
    def unavailable(cls) -> "HeadPose":
        return cls(yaw=0.0, pitch=0.0, roll=0.0, available=False)


def estimate(kps: Optional[np.ndarray]) -> HeadPose:
    """5 ta kalit nuqtadan bosh burilishini baholaydi.

    `kps` — (5, 2) shaklidagi massiv. Boshqa shakl yoki None kelsa
    `available=False` qaytariladi: taxmin qilishdan ko'ra "bilmayman" deyish
    xavfsizroq, chunki noto'g'ri yaw aktiv tekshiruvni aldab o'tkazishi mumkin.
    """
    if kps is None:
        return HeadPose.unavailable()

    points = np.asarray(kps, dtype=np.float64)
    if points.shape != (5, 2) or not np.all(np.isfinite(points)):
        return HeadPose.unavailable()

    left_eye = points[LEFT_EYE]
    right_eye = points[RIGHT_EYE]
    nose = points[NOSE]
    mouth_center = (points[LEFT_MOUTH] + points[RIGHT_MOUTH]) / 2.0

    eye_vector = right_eye - left_eye
    eye_distance = float(np.linalg.norm(eye_vector))
    if eye_distance < MIN_EYE_DISTANCE_PX:
        # Yuz juda kichik: piksel shovqini burun ofsetidan kattaroq bo'ladi.
        return HeadPose.unavailable()

    eye_center = (left_eye + right_eye) / 2.0

    # --- ROLL: ko'zlar chizig'ining gorizontalga nisbatan burchagi ---
    roll = math.degrees(math.atan2(float(eye_vector[1]), float(eye_vector[0])))

    # --- YAW: burun ko'zlar markazidan qancha siljigan ---
    #
    # Siljish ko'zlar CHIZIG'I bo'ylab o'lchanadi, oddiy X o'qi bo'ylab emas.
    # Aks holda boshni yonboshlatish (roll) yaw bo'lib ko'rinardi va
    # "chapga qarang" buyrug'ini boshni egish bilan aldab bo'lardi.
    eye_axis = eye_vector / eye_distance
    nose_offset = nose - eye_center
    along_eye_axis = float(np.dot(nose_offset, eye_axis))
    yaw = (along_eye_axis / eye_distance) * _YAW_SCALE_DEG

    # --- PITCH: burun ko'zlar va og'iz orasida qayerda ---
    #
    # Faqat ma'lumot uchun. Yuzning uch o'lchamli modelisiz bu qiymat odamdan
    # odamga sezilarli farq qiladi, shuning uchun qaror qabul qilishda
    # ishlatilmaydi.
    face_height = float(np.linalg.norm(mouth_center - eye_center))
    if face_height < 1e-6:
        pitch = 0.0
    else:
        perpendicular = np.array([-eye_axis[1], eye_axis[0]])
        along_vertical = float(np.dot(nose_offset, perpendicular))
        # 0.5 — burun ko'zlar va og'iz o'rtasida turgandagi nisbat.
        pitch = ((along_vertical / face_height) - 0.5) * _YAW_SCALE_DEG

    return HeadPose(
        yaw=_clamp(yaw, -90.0, 90.0),
        pitch=_clamp(pitch, -90.0, 90.0),
        roll=_clamp(roll, -90.0, 90.0),
        available=True,
    )


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))
