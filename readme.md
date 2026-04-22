# MLOps Exam 2026

## Branch: MLDLOPs-Exam2026

---

# Question 2: CityScape Image Segmentation Pipeline

## 1. Dataset Download

Dataset ko `gdown` command se download kiya aur unzip kiya:

```
gdown <dataset_link>
unzip dataset.zip
```

---

## 2. Dataset Details

* `data/CameraRGB/` → Input RGB images
* `data/CameraMask/` → Ground truth segmentation masks
* Total **23 classes** segmentation ke liye use kiye gaye

---

## 3. Train-Test Split & Dataloader

* Dataset ko **80%-20% split** kiya (seed = 42)
* Custom PyTorch Dataset aur DataLoader implement kiya

---

## 4. Model Training

* Model: **UNet**
* Input Channels: 3
* Output Classes: 23
* Epochs: 15 + 5 (fine-tuning)
* Loss Function: CrossEntropyLoss
* Optimizer: Adam

---

## 5. Training Curves

Training ke dauran:

* Loss decrease hua
* mIOU aur mDice increase hue

Generated plots:

* Training Loss Curve
* mIOU Curve
* mDice Curve

(Plots `training_plots.png` me available hain)

---

## 6. Test Set Results

Final evaluation on test dataset:

* **mIOU: 0.5127**
* **mDICE: 0.5693**

---

## 7. Required README Entry

**Question2: mIOU: 0.5127 and mDICE: 0.5693**

---

## 8. Streamlit App Deployment

### Page 1: Training Dashboard

* Training Loss Curve
* mIOU Curve
* mDice Curve
* Final Test Metrics

### Page 2: Inference

* User 4 test images upload karta hai
* App display karta hai:

  * Input Image
  * Ground Truth Mask
  * Predicted Mask

---

## 9. Screenshots

Screenshots GitHub repository me available hain:

```
Question2/screenshots/
```

* Page 1: Training Dashboard
* Page 2: Inference Results

---

## Final Conclusion

Model successfully train aur deploy kiya gaya using Streamlit.
Final performance metrics:

* mIOU > 0.48
* mDICE > 0.48

Isliye model **full evaluation criteria satisfy karta hai**.

---
