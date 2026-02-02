import pygame
import threading
import time
import random
import sys
import numpy as np

# ---------- CONFIG ----------
WIDTH, HEIGHT = 600, 600
BG_COLOR = (173, 216, 230)  # Light Blue (Awake)
SLEEP_BG = (10, 10, 15)  # Pitch Black (Sleep)
SLEEP_FACE = (20, 20, 25)

HEAD_COLOR = (255, 255, 255)
FACE_COLOR = (50, 50, 60)
CHEEK_COLOR = (255, 182, 193)
EYE_COLOR = (80, 255, 230)  # Cyan Neon


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
        self.font = pygame.font.SysFont("Comic Sans MS", 24, bold=True)
        self.small_font = pygame.font.SysFont("Arial", 18)
        self.z_font = pygame.font.SysFont("Arial", 40, bold=True)
        self.running = True
        self.state = "standby"
        self.mouth_open = 0
        self.blink_timer = time.time()
        self.is_blinking = False
        self.caption = "Booting..."
        self.command_queue = []
        self.is_recording = False
        self.last_interaction = time.time()
        self.particles = []

        # 🟢 NEW: Animation Offsets (Virtual Hardware)
        self.head_offset_x = 0
        self.head_offset_y = 0

    def reset_timer(self):
        self.last_interaction = time.time()
        if self.state == "sleeping":
            self.state = "idle"

    def set_state(self, state):
        if state != "sleeping":
            self.reset_timer()
        self.state = state

        # Caption Logic
        if state == "standby":
            self.caption = "Standby"
        elif state == "listening":
            self.caption = "Listening..."
        elif state == "thinking":
            self.caption = "Thinking..."
        elif state == "processing":
            self.caption = "Processing..."
        elif state == "idle":
            self.caption = "Ready"
        elif state == "sleeping":
            self.caption = "Zzz..."

    def set_caption(self, text):
        self.caption = text

    # 🟢 NEW: VIRTUAL NOD (YES)
    def trigger_nod(self):
        def animate():
            # Nod Down/Up 3 times
            for _ in range(3):
                # Down
                for i in range(10):
                    self.head_offset_y += 2
                    time.sleep(0.01)
                # Up
                for i in range(10):
                    self.head_offset_y -= 2
                    time.sleep(0.01)
            self.head_offset_y = 0

        threading.Thread(target=animate, daemon=True).start()

    # 🟢 NEW: VIRTUAL SHAKE (NO)
    def trigger_shake(self):
        def animate():
            # Shake Left/Right 3 times
            for _ in range(3):
                # Right
                for i in range(5):
                    self.head_offset_x += 4
                    time.sleep(0.01)
                # Left
                for i in range(10):
                    self.head_offset_x -= 4
                    time.sleep(0.01)
                # Center
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
                    # Mouth Movement
                    target = random.randint(10, 60)
                    self.mouth_open += (target - self.mouth_open) * 0.5

                    # 🟢 Talk Animation: Slight bobbing
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

        # 🟢 Head Position + Animation Offsets
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
            # Closed
            pygame.draw.line(self.screen, EYE_COLOR, (head_x + 100, eye_y + 40), (head_x + 160, eye_y + 40), 5)
            pygame.draw.line(self.screen, EYE_COLOR, (head_x + 240, eye_y + 40), (head_x + 300, eye_y + 40), 5)
        elif self.state == "thinking":
            # Googly
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
            # Normal
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

        # Text
        text_col = (200, 200, 200) if self.state in ["sleeping", "standby"] else (50, 50, 50)
        text_surf = self.font.render(self.caption, True, text_col)
        self.screen.blit(text_surf, text_surf.get_rect(center=(WIDTH // 2, HEIGHT - 80)))

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



