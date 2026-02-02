import pygame
import threading
import time
import random
import sys
import numpy as np
import os

# 🟢 بررسی نصب بودن کتابخانه‌های فارسی
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
except ImportError:
    print("❌ Error: Please run 'pip install python-bidi arabic-reshaper'")
    sys.exit(1)

# ---------- CONFIG ----------
WIDTH, HEIGHT = 600, 600
BG_COLOR = (173, 216, 230)  # Light Blue
SLEEP_BG = (10, 10, 15)  # Pitch Black
SLEEP_FACE = (20, 20, 25)

HEAD_COLOR = (255, 255, 255)
FACE_COLOR = (50, 50, 60)
CHEEK_COLOR = (255, 182, 193)
EYE_COLOR = (80, 255, 230)


# ---------- PARTICLE SYSTEM (ZZZ) ----------
class Particle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = random.randint(20, 40)
        self.alpha = 255
        self.speed = random.uniform(0.5, 1.5)
        self.wobble = random.uniform(-1, 1)

    def update(self):
        self.y -= self.speed
        self.x += np.sin(time.time() * 5 + self.wobble) * 0.5
        self.alpha -= 2
        self.size += 0.2

    def draw(self, surface, font):
        if self.alpha > 0:
            txt = font.render("Z", True, (255, 255, 255))
            txt.set_alpha(self.alpha)
            scaled = pygame.transform.scale(txt, (int(self.size), int(self.size)))
            surface.blit(scaled, (self.x, self.y))


# ---------- ROBOT UI ----------
class RobotUI:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN | pygame.SCALED)
        pygame.mouse.set_visible(False)
        pygame.display.set_caption("Robot")
        self.clock = pygame.time.Clock()

        # 🟢 1. لود کردن فونت فارسی اختصاصی
        # حتماً فایل persian_font.ttf را کنار برنامه بگذارید
        font_path = "persian_font.ttf"

        if not os.path.exists(font_path):
            print(f"⚠️ Warning: '{font_path}' not found! Trying system fonts...")
            # تلاش برای پیدا کردن فونت سیستم اگر فایل نبود
            font_path = pygame.font.match_font('arial')

        try:
            print(f"🔤 Loading Font from: {font_path}")
            self.font = pygame.font.Font(font_path, 28)
            self.question_font = pygame.font.Font(font_path, 24)
            self.z_font = pygame.font.Font(font_path, 40)
        except Exception as e:
            print(f"❌ Font Load Error: {e}")
            print("Using default font (Persian might look like blocks)")
            self.font = pygame.font.SysFont("Arial", 28, bold=True)
            self.question_font = pygame.font.SysFont("Arial", 24, bold=True)
            self.z_font = pygame.font.SysFont("Arial", 40, bold=True)

        self.running = True
        self.state = "standby"
        self.mouth_open = 0
        self.blink_timer = time.time()
        self.is_blinking = False

        self.caption = "Booting..."
        self.user_text = ""

        self.command_queue = []
        self.is_recording = False
        self.last_interaction = time.time()
        self.particles = []

        self.head_offset_x = 0
        self.head_offset_y = 0

    # 🟢 2. تابع اصلاح متن فارسی
    def render_persian(self, text):
        if not text: return ""
        try:
            reshaped_text = arabic_reshaper.reshape(text)
            bidi_text = get_display(reshaped_text)
            return bidi_text
        except:
            return text

    def reset_timer(self):
        self.last_interaction = time.time()
        if self.state == "sleeping":
            self.state = "idle"

    def set_state(self, state):
        if state != "sleeping":
            self.reset_timer()
        self.state = state

        if state == "standby":
            self.caption = "آماده‌باش"
        elif state == "listening":
            self.caption = "گوش می‌دهم..."
            self.user_text = ""
        elif state == "thinking":
            self.caption = "فکر می‌کنم..."
        elif state == "processing":
            self.caption = "پردازش..."
        elif state == "idle":
            self.caption = "آماده"
        elif state == "sleeping":
            self.caption = "خواب..."

    def set_caption(self, text):
        self.caption = text

    def set_user_question(self, text):
        self.user_text = f"شما: {text}"

    def trigger_nod(self):
        def animate():
            for _ in range(3):
                for i in range(10):
                    self.head_offset_y += 2
                    time.sleep(0.01)
                for i in range(10):
                    self.head_offset_y -= 2
                    time.sleep(0.01)
            self.head_offset_y = 0

        threading.Thread(target=animate, daemon=True).start()

    def trigger_shake(self):
        def animate():
            for _ in range(3):
                for i in range(5):
                    self.head_offset_x += 4
                    time.sleep(0.01)
                for i in range(10):
                    self.head_offset_x -= 4
                    time.sleep(0.01)
                for i in range(5):
                    self.head_offset_x += 4
                    time.sleep(0.01)
            self.head_offset_x = 0

        threading.Thread(target=animate, daemon=True).start()

    def play_file(self, filename):
        def play_thread():
            try:
                self.state = "talking"
                pygame.mixer.music.load(filename)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    target = random.randint(10, 60)
                    self.mouth_open += (target - self.mouth_open) * 0.5
                    self.head_offset_y = np.sin(time.time() * 15) * 2
                    time.sleep(0.1)

                if self.state == "talking":
                    self.state = "idle"
            except:
                pass
            finally:
                self.mouth_open = 0
                self.head_offset_y = 0

        threading.Thread(target=play_thread, daemon=True).start()

    def draw_rounded_rect(self, surface, color, rect, radius=20):
        pygame.draw.rect(surface, color, rect, border_radius=radius)

    def draw(self):
        if self.state == "standby":
            bg = (5, 5, 10)
            current_face_color = (20, 20, 30)
        elif self.state == "sleeping":
            bg = SLEEP_BG
            current_face_color = SLEEP_FACE
        else:
            bg = BG_COLOR
            current_face_color = FACE_COLOR

        self.screen.fill(bg)

        if self.state in ["sleeping", "standby"]:
            bounce = 20
        else:
            bounce = np.sin(time.time() * 2) * 5

        head_x = 100 + self.head_offset_x
        head_y = 100 + bounce + self.head_offset_y

        # Antenna
        pygame.draw.line(self.screen, (100, 100, 100), (300, head_y), (300, head_y - 40), 5)
        ant_color = (255, 50, 50) if self.state == "listening" else (200, 200, 200)
        pygame.draw.circle(self.screen, ant_color, (300, int(head_y - 40)), 15)

        self.draw_rounded_rect(self.screen, HEAD_COLOR, (head_x, head_y, 400, 350), 60)
        self.draw_rounded_rect(self.screen, current_face_color, (head_x + 40, head_y + 50, 320, 250), 40)

        # Eyes
        eye_y = head_y + 120
        if self.state in ["standby", "sleeping"]:
            pygame.draw.line(self.screen, EYE_COLOR, (head_x + 100, eye_y + 40), (head_x + 160, eye_y + 40), 5)
            pygame.draw.line(self.screen, EYE_COLOR, (head_x + 240, eye_y + 40), (head_x + 300, eye_y + 40), 5)
        elif self.state == "thinking":
            t = time.time() * 10
            off_x1 = np.sin(t) * 15;
            off_y1 = np.cos(t) * 15
            off_x2 = np.sin(t + 2) * 15;
            off_y2 = np.cos(t + 2) * 15
            pygame.draw.circle(self.screen, (255, 255, 255), (int(head_x + 130), int(eye_y + 40)), 40)
            pygame.draw.circle(self.screen, (0, 0, 0), (int(head_x + 130 + off_x1), int(eye_y + 40 + off_y1)), 15)
            pygame.draw.circle(self.screen, (255, 255, 255), (int(head_x + 270), int(eye_y + 40)), 40)
            pygame.draw.circle(self.screen, (0, 0, 0), (int(head_x + 270 + off_x2), int(eye_y + 40 + off_y2)), 15)
        else:
            now = time.time()
            if now - self.blink_timer > random.uniform(3, 6):
                self.is_blinking = True
                if now - self.blink_timer > random.uniform(3, 6) + 0.15:
                    self.is_blinking = False;
                    self.blink_timer = now
            if self.is_blinking:
                pygame.draw.line(self.screen, EYE_COLOR, (head_x + 100, eye_y + 40), (head_x + 160, eye_y + 40), 8)
                pygame.draw.line(self.screen, EYE_COLOR, (head_x + 240, eye_y + 40), (head_x + 300, eye_y + 40), 8)
            else:
                pygame.draw.ellipse(self.screen, EYE_COLOR, (head_x + 100, eye_y, 60, 80))
                pygame.draw.ellipse(self.screen, EYE_COLOR, (head_x + 240, eye_y, 60, 80))
                pygame.draw.circle(self.screen, (255, 255, 255), (int(head_x + 120), int(eye_y + 20)), 10)
                pygame.draw.circle(self.screen, (255, 255, 255), (int(head_x + 260), int(eye_y + 20)), 10)

        # Mouth
        pygame.draw.circle(self.screen, CHEEK_COLOR, (head_x + 80, int(head_y + 200)), 20)
        pygame.draw.circle(self.screen, CHEEK_COLOR, (head_x + 320, int(head_y + 200)), 20)
        mouth_x, mouth_y = head_x + 200, head_y + 220

        if self.state == "talking":
            h = max(10, self.mouth_open)
            pygame.draw.ellipse(self.screen, (255, 255, 255), (mouth_x - 40, mouth_y - h / 2, 80, h))
        elif self.state == "thinking":
            pygame.draw.circle(self.screen, (255, 255, 255), (mouth_x, int(mouth_y)), 15)
        elif self.state in ["sleeping", "standby"]:
            pygame.draw.line(self.screen, (200, 200, 200), (mouth_x - 10, mouth_y), (mouth_x + 10, mouth_y), 3)
        else:
            pygame.draw.arc(self.screen, (255, 255, 255), (mouth_x - 40, mouth_y - 20, 80, 40), 3.14, 6.28, 5)

        # Zzz Particles
        if self.state == "sleeping":
            if random.randint(0, 30) == 0: self.particles.append(Particle(350, head_y))
        for p in self.particles[:]:
            p.update()
            p.draw(self.screen, self.z_font)
            if p.alpha <= 0: self.particles.remove(p)

        # 🟢 3. رسم متن‌ها با اصلاح فارسی
        text_col = (200, 200, 200) if self.state in ["sleeping", "standby"] else (50, 50, 50)

        # Caption
        fixed_caption = self.render_persian(self.caption)
        text_surf = self.font.render(fixed_caption, True, text_col)
        self.screen.blit(text_surf, text_surf.get_rect(center=(WIDTH // 2, HEIGHT - 120)))

        # User Question
        if self.user_text:
            fixed_user = self.render_persian(self.user_text)
            q_surf = self.question_font.render(fixed_user, True, (0, 0, 80))
            self.screen.blit(q_surf, q_surf.get_rect(center=(WIDTH // 2, HEIGHT - 60)))

        pygame.display.flip()

    def run(self):
        while self.running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self.running = False
                elif e.type == pygame.KEYDOWN:
                    self.reset_timer()
                    if e.key == pygame.K_ESCAPE: self.running = False
            self.draw()
            self.clock.tick(60)
        pygame.quit()