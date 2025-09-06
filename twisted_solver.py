import numpy as np
from itertools import product
import string

def conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def rotate(v, q):
    return mul(mul(q, v), conj(q))

# Data từ output.txt
ar = [
    np.array([  17.33884894,   81.37080239, -143.96234736,  123.95164171]),
    np.array([ 168.34743674,  100.91788802, -135.90959582,  146.37617105]),
    np.array([ 157.94860314,   49.20197906, -155.2459834 ,   73.56498047]),
    np.array([   9.1131532 ,   49.36829422, -117.25335109,  181.11592151]),
    np.array([ 223.96684757,  -12.0765699 , -126.07584525,  125.88335439]),
    np.array([ 80.13452478,  40.78304285, -51.15180044, 143.18760932]),
    np.array([ 251.41332497,   48.04296984, -128.92087521,   68.4732401 ]),
    np.array([108.94539496,  -0.41865393, -53.94228136, 100.98194223]),
    np.array([183.06845007,  27.56200727, -52.57316992,  44.05723383]),
    np.array([ 96.56452698,  60.67582903, -76.44584757,  40.88253203])
]

def solve_with_known_format():
    """
    Giải challenge với biết format flag: ictf{
    """
    print("Solving twisted challenge with format 'ictf{'...")
    
    # Flag format: ictf{...}
    # Chunk 0: "ict"
    # Chunk 1: "f{" + padding
    # Các chunk còn lại: nội dung flag
    
    known_chunks = {
        0: "ict",
        1: "f{0"  # f{ + padding "0"
    }
    
    # Brute force để tìm các tham số transform
    def find_transform_params(target_chunk, known_text):
        """Tìm tham số transform cho chunk đã biết"""
        best_params = None
        best_score = float('inf')
        
        # Thử nhiều giá trị w
        for w in np.linspace(0, 255, 100):
            original_q = np.array([w, ord(known_text[0]), ord(known_text[1]), ord(known_text[2])])
            
            # Thử nhiều quaternion rotation
            np.random.seed(42)  # For reproducible results
            for _ in range(1000):
                r = np.random.randn(4)
                r /= np.linalg.norm(r)
                
                # Forward transform
                rotated = rotate(original_q, r)
                
                # Random affine transform parameters
                a = np.linalg.qr(np.random.randn(3, 3))[0]
                if np.linalg.det(a) < 0:
                    a[:, 0] = -a[:, 0]
                s = 1.25
                t = np.random.uniform(-90, 90, size=3)
                
                vec = rotated[1:]
                transformed_vec = s * (a @ vec) + t
                result = np.concatenate(([rotated[0]], transformed_vec))
                
                # So sánh với target
                diff = np.linalg.norm(result - target_chunk)
                if diff < best_score:
                    best_score = diff
                    best_params = {
                        'w': w,
                        'r': r,
                        'a': a,
                        's': s,
                        't': t,
                        'original_q': original_q,
                        'rotated': rotated
                    }
                    
                    if diff < 0.5:  # Good enough match
                        print(f"Found good match for '{known_text}': score = {diff:.4f}")
                        return best_params
        
        print(f"Best match for '{known_text}': score = {best_score:.4f}")
        return best_params
    
    # Tìm transform params từ chunk đã biết
    print("Finding transform parameters from known chunks...")
    params_0 = find_transform_params(ar[0], "ict")
    params_1 = find_transform_params(ar[1], "f{0")
    
    if params_0 is None and params_1 is None:
        print("Could not find good transform parameters")
        return None
    
    # Sử dụng params tốt nhất
    best_params = params_0 if params_0 and params_0 else params_1
    
    print(f"Using transform parameters with score")
    
    # Giải các chunk còn lại
    flag_parts = ["ict", "f{"
    
    for chunk_idx in range(2, len(ar)):
        print(f"Solving chunk {chunk_idx}...")
        best_chunk = ""
        best_score = float('inf')
        
        # Thử tất cả combination 3 ký tự có thể
        possible_chars = string.ascii_lowercase + string.digits + "_!@#$%^&*()[]"
        
        for chars in product(possible_chars, repeat=3):
            chunk_text = ''.join(chars)
            
            # Thử với w tương tự
            w = best_params['w']
            original_q = np.array([w, ord(chunk_text[0]), ord(chunk_text[1]), ord(chunk_text[2])])
            
            # Sử dụng transform params đã tìm được
            rotated = rotate(original_q, best_params['r'])
            vec = rotated[1:]
            transformed_vec = best_params['s'] * (best_params['a'] @ vec) + best_params['t']
            result = np.concatenate(([rotated[0]], transformed_vec))
            
            # So sánh
            diff = np.linalg.norm(result - ar[chunk_idx])
            if diff < best_score:
                best_score = diff
                best_chunk = chunk_text
        
        print(f"Best chunk {chunk_idx}: '{best_chunk}' (score: {best_score:.4f})")
        flag_parts.append(best_chunk)
    
    # Reconstruct flag
    flag = ''.join(flag_parts)
    # Remove padding zeros and add closing brace
    flag = flag.replace('0', '').rstrip('0') + '}'
    
    return flag

def solve_statistical_approach():
    """
    Approach thống kê: phân tích range giá trị để đoán ký tự
    """
    print("\nStatistical analysis of chunks:")
    
    for i, chunk_data in enumerate(ar):
        print(f"Chunk {i}: w={{chunk_data[0]:.2f}}, vec=[{{chunk_data[1]:.2f}}, {{chunk_data[2]:.2f}}, {{chunk_data[3]:.2f}}]")
        
        # Estimate possible ASCII values
        # Sau transform, giá trị gốc có thể trong khoảng nào?
        vec = chunk_data[1:]
        estimated_range = [np.min(vec) / 1.25 - 90, np.max(vec) / 1.25 + 90]
        print(f"  Estimated original range: [{{estimated_range[0]:.1f}}, {{estimated_range[1]:.1f}}]")
        
        # Guess possible characters based on ASCII range
        possible_chars = []
        for ascii_val in range(32, 127):  # Printable ASCII
            if estimated_range[0] <= ascii_val <= estimated_range[1]:
                possible_chars.append(chr(ascii_val))
        
        print(f"  Possible chars: {{possible_chars[:20]}}...")  # Show first 20

if __name__ == "__main__":
    print("=" * 50)
    print("TWISTED CHALLENGE SOLVER")
    print("Flag format: ictf{...}")
    print("=" * 50)
    
    # Statistical analysis first
    solve_statistical_approach()
    
    print("\n" + "=" * 50)
    print("ATTEMPTING TO SOLVE...")
    print("=" * 50)
    
    # Main solving
    flag = solve_with_known_format()
    
    if flag:
        print(f"\n🎉 POTENTIAL FLAG: {{flag}}")
    else:
        print("\n❌ Could not solve the challenge")
        
    print("\nIf the above doesn't work, try:")
    print("1. Adjust the threshold values")
    print("2. Try more random seeds")
    print("3. Expand the character set")
    print("4. Use different noise estimation")
