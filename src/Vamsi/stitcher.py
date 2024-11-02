import cv2
import numpy as np
import glob
import os

class PanaromaStitcher:
    def __init__(self):
        self.debug_mode = False
    
    def project_cylindrical(self, img, f):
        h, w = img.shape[:2]
        out = np.zeros_like(img)
        
        cx = w // 2
        cy = h // 2
        
        for x in range(w):
            for y in range(h):
                t = (x - cx) / f
                v = (y - cy) / f
                
                nx = np.sin(t)
                ny = v
                nz = np.cos(t)
                
                px = int(f * nx / nz + cx)
                py = int(f * ny / nz + cy)
                
                if 0 <= px < w and 0 <= py < h:
                    out[y, x] = img[py, px]
        
        return out

    def get_homography(self, p1, p2):
        m = []
        
        for i in range(len(p1)):
            x, y = p1[i][0], p1[i][1]
            u, v = p2[i][0], p2[i][1]
            
            r1 = [x, y, 1, 0, 0, 0, -u * x, -u * y, -u]
            r2 = [0, 0, 0, x, y, 1, -v * x, -v * y, -v]
            
            m.extend([r1, r2])
            
        m = np.array(m)
        _, _, v = np.linalg.svd(m)
        h = v[-1].reshape((3, 3))
        
        return h

    def ransac_homography(self, p1, p2, max_iter=2000):
        thresh = 5
        best_h = None
        best_inliers = []
        max_inliers = 0
        
        for _ in range(max_iter):
            idx = np.random.choice(p1.shape[0], 4, replace=False)
            h = self.get_homography(p1[idx], p2[idx])
            
            pts = np.hstack([p1, np.ones((p1.shape[0], 1))])
            xf = np.dot(h, pts.T).T
            pp = xf[:, :2] / xf[:, 2].reshape(-1, 1)
            
            err = np.linalg.norm(p2 - pp, axis=1)
            inliers = np.where(err < thresh)[0]
            
            if len(inliers) > max_inliers:
                max_inliers = len(inliers)
                best_h = h
                best_inliers = inliers

        if len(best_inliers) > 4:
            best_h = self.get_homography(p1[best_inliers], p2[best_inliers])
            
        return best_h, best_inliers

    def match_images(self, img1, img2):
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        
        flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        matches = flann.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good) < 4:
            print("Not enough matches")
            return None

        src = np.float32([kp1[m.queryIdx].pt for m in good])
        dst = np.float32([kp2[m.trainIdx].pt for m in good])
        
        return self.ransac_homography(src, dst)

    def compute_homographies(self, imgs, mid):
        H = [np.eye(3)]
        
        # Left side
        for i in range(mid - 1, -1, -1):
            res = self.match_images(imgs[i], imgs[i + 1])
            if res is not None:
                H.insert(0, res[0] @ H[0])
        
        # Right side
        for i in range(mid + 1, len(imgs)):
            res = self.match_images(imgs[i - 1], imgs[i])
            if res is not None:
                H.append(np.linalg.inv(res[0]) @ H[-1])
                
        return H

    def get_output_dimensions(self, imgs, H):
        corners = []
        for i, img in enumerate(imgs):
            h, w = img.shape[:2]
            pts = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]])
            warp = (H[i] @ pts.T).T
            proj = warp[:, :2] / warp[:, 2:3]
            corners.extend(proj)
        
        corners = np.array(corners)
        return np.min(corners, axis=0).astype(int), np.max(corners, axis=0).astype(int)

    def blend_images(self, imgs, H, out_size, mid):
        pano = cv2.warpPerspective(imgs[mid], H[mid], out_size)
        
        for i, img in enumerate(imgs):
            if i != mid:
                warp = cv2.warpPerspective(img, H[i], out_size)
                mask = warp > 0
                pano[mask] = warp[mask]
                
        return pano

    def make_panaroma_for_images_in(self, path):
        # Load and project images
        imgs = [cv2.imread(f) for f in sorted(glob.glob(path + os.sep + '*'))]
        mid = len(imgs) // 2
        f = imgs[mid].shape[1] / (2 * np.pi) * 8
        cyl_imgs = [self.project_cylindrical(img, f) for img in imgs]
        
        # Get homographies
        H = self.compute_homographies(cyl_imgs, mid)
        
        # Calculate panorama size
        min_xy, max_xy = self.get_output_dimensions(cyl_imgs, H)
        
        # Apply offset
        T = np.array([[1, 0, -min_xy[0]], [0, 1, -min_xy[1]], [0, 0, 1]])
        H_final = [T @ h for h in H]
        
        # Create final panorama
        out_size = (max_xy[0] - min_xy[0], max_xy[1] - min_xy[1])
        pano = self.blend_images(cyl_imgs, H_final, out_size, mid)
        
        return pano, H