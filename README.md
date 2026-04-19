# InfraScan Sentinel 🏢🏛️

System works partially
Main issue: inconsistent edge localization due to coordinate scaling bug
Next step: fix depth-space indexing and validate depth-based scoring

<img width="674" height="721" alt="image" src="https://github.com/user-attachments/assets/1f5d56b6-4f79-4d04-a7c7-5b405d0f1ecb" />



### **Subtitle: Non-Contact Seismic Gap Analysis for High-Density Urban Environments**

## **Overview**
In high-density cities like **Kathmandu**, buildings are often constructed with insufficient separation. During a seismic event, these adjacent structures can collide—a phenomenon known as **Seismic Pounding**.

This project utilizes **Computer Vision (OpenCV)** and **Structural Dynamics Theory** to identify at-risk building clusters by measuring real-world gaps through non-invasive video analysis.

## **The Problem**
According to **Nepal National Building Code (NBC) 105:2020**, structural separation is mandatory to prevent collision. However, thousands of existing structures in the Kathmandu Valley lack the required **"Seismic Gap."** Traditional measurement methods are slow and labor-intensive.

## **Technical Implementation**
* **Feature Extraction:** Utilizing Canny Edge Detection and **Hough Line Transforms** to isolate vertical structural boundaries.
* **Metrology:** Mapping pixel-width to real-world centimeters using camera intrinsic calibration.
* **Risk Assessment:** Categorizing gaps based on height-to-separation ratios defined by seismic safety standards.

## **Local-to-Global Impact**
While developed in the high-seismic context of Nepal, the **InfraScan-Sentinel** logic is applicable to aging infrastructure in NYC, Tokyo, and Mexico City, where "pounding" remains a hidden threat to urban resilience.
