"""
Pix2Pix Training Script — Edges2Shoes
COMP 484 | Kathmandu University

Usage:
    python train.py --dataset ./edges2shoes --epochs 50 --output ./output
    python train.py --dataset ./edges2shoes --epochs 50 --resume ./output/checkpoints
"""

import os, argparse, numpy as np, matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def parse_args():
    p = argparse.ArgumentParser(description="Train Pix2Pix (sketch->photo)")
    p.add_argument("--dataset",    default="./edges2shoes")
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch_size", type=int,   default=16)
    p.add_argument("--lambda_l1",  type=int,   default=100)
    p.add_argument("--lr",         type=float, default=2e-4)
    p.add_argument("--output",     default="./output")
    p.add_argument("--resume",     default=None)
    return p.parse_args()

IMG_H = IMG_W = 256
INIT  = tf.random_normal_initializer(0.0, 0.02)

def load_image_pair(path):
    raw = tf.io.read_file(path)
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.cast(img, tf.float32)
    w   = tf.shape(img)[1] // 2
    return img[:, :w, :], img[:, w:, :]

def normalize(inp, real):
    return (inp / 127.5) - 1.0, (real / 127.5) - 1.0

def augment(inp, real):
    stack = tf.stack([tf.image.resize(inp,  [286, 286], method="nearest"),
                      tf.image.resize(real, [286, 286], method="nearest")], axis=0)
    crop  = tf.image.random_crop(stack, [2, IMG_H, IMG_W, 3])
    inp, real = crop[0], crop[1]
    if tf.random.uniform(()) > 0.5:
        inp  = tf.image.flip_left_right(inp)
        real = tf.image.flip_left_right(real)
    return inp, real

def build_dataset(folder, batch_size, train=True):
    ds = tf.data.Dataset.list_files(os.path.join(folder, "*.jpg"), shuffle=train)
    def pt(f):
        i,r = load_image_pair(f); i,r = augment(i,r); return normalize(i,r)
    def pv(f):
        i,r = load_image_pair(f)
        i = tf.image.resize(i,[IMG_H,IMG_W]); r = tf.image.resize(r,[IMG_H,IMG_W])
        return normalize(i,r)
    return (ds.map(pt if train else pv, num_parallel_calls=tf.data.AUTOTUNE)
              .batch(batch_size).prefetch(tf.data.AUTOTUNE))

def downsample(f, apply_bn=True):
    b = keras.Sequential([layers.Conv2D(f,4,strides=2,padding="same",use_bias=False,kernel_initializer=INIT)])
    if apply_bn: b.add(layers.BatchNormalization())
    b.add(layers.LeakyReLU(0.2)); return b

def upsample(f, dropout=False):
    b = keras.Sequential([layers.Conv2DTranspose(f,4,strides=2,padding="same",use_bias=False,kernel_initializer=INIT),
                          layers.BatchNormalization()])
    if dropout: b.add(layers.Dropout(0.5))
    b.add(layers.ReLU()); return b

def build_generator():
    inp  = layers.Input(shape=[IMG_H, IMG_W, 3])
    encs = [downsample(64,False), downsample(128), downsample(256), downsample(512),
            downsample(512), downsample(512), downsample(512), downsample(512)]
    decs = [upsample(512,True), upsample(512,True), upsample(512,True), upsample(512),
            upsample(256), upsample(128), upsample(64)]
    last = layers.Conv2DTranspose(3,4,strides=2,padding="same",kernel_initializer=INIT,activation="tanh")
    x = inp; skips=[]
    for e in encs: x=e(x); skips.append(x)
    for d,s in zip(decs, reversed(skips[:-1])): x=d(x); x=layers.Concatenate()([x,s])
    return keras.Model(inputs=inp, outputs=last(x), name="Generator")

def build_discriminator():
    sk=layers.Input([IMG_H,IMG_W,3],name="sketch"); ph=layers.Input([IMG_H,IMG_W,3],name="photo")
    x=layers.Concatenate()([sk,ph])
    x=downsample(64,False)(x); x=downsample(128)(x); x=downsample(256)(x)
    x=layers.ZeroPadding2D()(x)
    x=layers.Conv2D(512,4,strides=1,use_bias=False,kernel_initializer=INIT)(x)
    x=layers.BatchNormalization()(x); x=layers.LeakyReLU(0.2)(x); x=layers.ZeroPadding2D()(x)
    x=layers.Conv2D(1,4,strides=1,kernel_initializer=INIT)(x)
    return keras.Model(inputs=[sk,ph], outputs=x, name="Discriminator")

BCE = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def gen_loss(df,go,tgt,lam): g=BCE(tf.ones_like(df),df); l=tf.reduce_mean(tf.abs(tgt-go)); return g+lam*l,g,l
def disc_loss(dr,df): return BCE(tf.ones_like(dr),dr)+BCE(tf.zeros_like(df),df)

def make_step(gen,disc,go,do,lam):
    @tf.function
    def step(inp,tgt):
        with tf.GradientTape() as gt, tf.GradientTape() as dt:
            fake=gen(inp,training=True); dr=disc([inp,tgt],training=True); df=disc([inp,fake],training=True)
            gt_,gg,gl=gen_loss(df,fake,tgt,lam); dt_=disc_loss(dr,df)
        go.apply_gradients(zip(gt.gradient(gt_,gen.trainable_variables),gen.trainable_variables))
        do.apply_gradients(zip(dt.gradient(dt_,disc.trainable_variables),disc.trainable_variables))
        return {"gen":gt_,"gan":gg,"l1":gl,"disc":dt_}
    return step

def denorm(img): return np.clip((img+1.)/2.,0,1)

def save_preview(gen,sk,ph,epoch,d):
    pred=gen(sk[:1],training=False)
    fig,ax=plt.subplots(1,3,figsize=(12,4)); fig.suptitle(f"Epoch {epoch}",fontweight="bold")
    for a,img,t in zip(ax,[sk[0],pred[0],ph[0]],["Sketch","Generated","Ground Truth"]):
        a.imshow(denorm(img.numpy())); a.set_title(t); a.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(d,f"epoch_{epoch:03d}.png"),dpi=120); plt.close()

def plot_losses(history, out_dir):
    fig,(a1,a2)=plt.subplots(1,2,figsize=(12,4)); fig.suptitle("Training Loss",fontweight="bold")
    a1.plot(history["gen"],label="Gen Total",linewidth=2)
    a1.plot(history["gan"],label="GAN",linestyle="--")
    a1.plot(history["l1"], label="L1 (×λ)",linestyle=":")
    a1.set_title("Generator"); a1.legend(); a1.grid(alpha=0.3)
    a2.plot(history["disc"],label="Discriminator",color="crimson",linewidth=2)
    a2.set_title("Discriminator"); a2.legend(); a2.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,"loss_curves.png"),dpi=120); plt.close()

def main():
    args=parse_args()
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu,True)
    print(f"TF {tf.__version__}  GPU: {tf.config.list_physical_devices('GPU') or 'CPU only'}")
    img_dir  = os.path.join(args.output,"previews")
    ckpt_dir = os.path.join(args.output,"checkpoints")
    os.makedirs(img_dir,exist_ok=True); os.makedirs(ckpt_dir,exist_ok=True)
    train_ds = build_dataset(os.path.join(args.dataset,"train"),args.batch_size)
    print(f"Train batches: {len(train_ds)}")
    gen=build_generator(); disc=build_discriminator()
    print(f"Gen params: {gen.count_params():,}  |  Disc params: {disc.count_params():,}")
    go=tf.keras.optimizers.Adam(args.lr,beta_1=0.5); do=tf.keras.optimizers.Adam(args.lr,beta_1=0.5)
    ckpt=tf.train.Checkpoint(gen=gen,disc=disc,go=go,do=do)
    mgr=tf.train.CheckpointManager(ckpt,ckpt_dir,max_to_keep=2)
    start=0
    if args.resume:
        latest=tf.train.latest_checkpoint(args.resume)
        if latest:
            ckpt.restore(latest)
            try: start=int(latest.split("-")[-1])*5
            except: pass
            print(f"Resumed from {latest} (epoch ~{start})")
    sk_sample,ph_sample=next(iter(train_ds))
    step=make_step(gen,disc,go,do,args.lambda_l1)
    hist={"gen":[],"gan":[],"l1":[],"disc":[]}
    loss_log=os.path.join(args.output,"loss_history.txt")
    print(f"\nTraining epochs {start+1} → {args.epochs}\n{'='*60}")
    for ep in range(start+1,args.epochs+1):
        s={"gen":0.,"gan":0.,"l1":0.,"disc":0.}; n=0
        for i,t in train_ds:
            l=step(i,t)
            for k in s: s[k]+=l[k].numpy()
            n+=1
        avg={k:v/n for k,v in s.items()}
        for k in hist: hist[k].append(avg[k])
        print(f"  Epoch [{ep:3d}/{args.epochs}]  Gen={avg['gen']:.4f}  GAN={avg['gan']:.4f}  L1={avg['l1']:.4f}  Disc={avg['disc']:.4f}")
        with open(loss_log,"a") as f:
            f.write(f"Epoch {ep:3d} | Gen={avg['gen']:.4f} | GAN={avg['gan']:.4f} | L1={avg['l1']:.4f} | Disc={avg['disc']:.4f}\n")
        if ep%5==0:
            save_preview(gen,sk_sample,ph_sample,ep,img_dir)
            mgr.save(); gen.save(os.path.join(args.output,"pix2pix_generator.keras"))
            print(f"    ✓ checkpoint + model saved (epoch {ep})")
    plot_losses(hist,args.output)
    gen.save(os.path.join(args.output,"pix2pix_generator_final.keras"))
    print(f"\n✓ Done. Final model → {args.output}/pix2pix_generator_final.keras")

if __name__=="__main__": main()
# data pipeline
# architecture blocks
# generator defined
# discriminator
