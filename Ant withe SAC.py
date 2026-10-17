# تثبيت خادم العرض الوهمي
!apt-get update && apt-get install -y xvfb

# تثبيت المكتبات المعتمدة (النسخ الحديثة تدعم gymnasium)
!pip install stable-baselines3[extra] gymnasium[mujoco] pyvirtualdisplay moviepy


import os
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

# 1. إعداد مجلدات الحفظ التلقائي
log_dir = "./ant_sac_logs/"
checkpoint_dir = "./ant_checkpoints/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# 2. إنشاء البيئة وتغليفها بالمراقب
env = gym.make("Ant-v4")
env = Monitor(env, log_dir)

# 3. إعداد "درع الحماية" (حفظ نسخة من الدماغ كل 100,000 خطوة)
checkpoint_callback = CheckpointCallback(
    save_freq=100000, 
    save_path=checkpoint_dir,
    name_prefix='sac_ant_backup'  # سيحفظ الملفات باسم sac_ant_backup_100000_steps.zip وهكذا
)

print("🚀 بدء تدريب النملة (Ant-v4) مع تفعيل نظام الحماية ضد انقطاع الإنترنت...")

# 4. بناء النموذج بالمعاملات الذهبية لـ SAC
model = SAC(
    "MlpPolicy",
    env,
    learning_rate=0.0003,
    buffer_size=1000000,
    batch_size=256,
    ent_coef="auto",
    gamma=0.99,
    tau=0.005,
    train_freq=1,
    gradient_steps=1,
    learning_starts=10000,
    verbose=1
)

# 5. التدريب المستمر (مع تمرير درع الحماية Callback)
# سيستغرق التدريب لمليون خطوة حوالي 30-45 دقيقة.
model.learn(total_timesteps=300_000, log_interval=10, callback=checkpoint_callback)

# 6. الحفظ النهائي وإغلاق البيئة (إذا اكتمل التدريب بسلام)
model.save("sac_ant_expert_final")
env.close()

print("✅ اكتمل التدريب للمليون خطوة وتم حفظ النموذج النهائي بنجاح!")


# ==========================================
# تصوير الفيديو للتقييم بعد التطور
# ==========================================

import os
import imageio
from IPython.display import Video, display
from pyvirtualdisplay import Display
from stable_baselines3 import SAC
import gymnasium as gym

# 1. تشغيل الشاشة الوهمية
virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()

# 2. الاستدعاء الصريح وإصلاح سرعة الإطارات (Ant تحتاج 20 إطار في الثانية فقط)
from gymnasium.envs.mujoco.ant_v4 import AntEnv
if "render_fps" in AntEnv.metadata:
    AntEnv.metadata["render_fps"] = 20  # تم التصحيح هنا

# 3. إنشاء البيئة
env_test = gym.make("Ant-v4", render_mode="rgb_array")

# 4. تحميل دماغ النملة (الجزء الأول - 100 ألف خطوة)
model = SAC.load("/kaggle/working/sac_ant_expert_final.zip")

frames = []
state, _ = env_test.reset()
done = False
steps = 0

print("🎥 بدء تسجيل النملة وهي تمشي...")

# 5. حلقة التصوير
while not done and steps < 1000:
    frames.append(env_test.render())
    
    # اتخاذ القرار الحتمي
    action, _states = model.predict(state, deterministic=True)
    state, reward, terminated, truncated, _ = env_test.step(action)
    
    done = terminated or truncated
    steps += 1

env_test.close()

# 6. حفظ الفيديو بسرعة 20 إطار في الثانية
video_path = './mujoco_ant_sac.mp4'
imageio.mimsave(video_path, frames, fps=20, macro_block_size=None)  # تم التصحيح هنا أيضاً
print(f"✅ تم تسجيل الفيديو بنجاح! إجمالي الإطارات: {len(frames)}")

display(Video(video_path, embed=True))