import numpy as np
from scipy.optimize import minimize
import itertools

class QuaternionCTFSolver:
    def __init__(self, data_points):
        self.ar = data_points
        self.sigma = 0.22  # Noise level from original code
        self.s = 1.25     # Scale factor from original code
        
    def conj(self, q):
        """Quaternion conjugate"""
        return np.array([q[0], -q[1], -q[2], -q[3]])
    
    def mul(self, q1, q2):
        """Quaternion multiplication"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def rotate(self, v, q):
        """Apply quaternion rotation to vector"""
        return self.mul(self.mul(q, v), self.conj(q))
    
    def inverse_rotate(self, v, q):
        """Apply inverse quaternion rotation"""
        return self.rotate(v, self.conj(q))
    
    def analyze_structure(self):
        """Analyze the mathematical structure of the challenge"""
        print("QUATERNION CTF CHALLENGE ANALYSIS")
        print("=" * 50)
        print(f"Number of data points: {len(self.ar)}")
        print(f"Total flag characters: {len(self.ar) * 3}")
        print(f"Expected flag format: ictf{{<{{len(self.ar) * 3 - 6}}_chars>}}")
        
        # Analyze data point ranges
        all_values = np.concatenate(self.ar)
        print(f"\nData value ranges:")
        print(f"Min: {np.min(all_values):.2f}")
        print(f"Max: {np.max(all_values):.2f}")
        print(f"Mean: {np.mean(all_values):.2f}")
        print(f"Std: {np.std(all_values):.2f}")
        
        # Analyze each component separately
        for i in range(4):
            component_values = [data[i] for data in self.ar]
            print(f"Component {i}: min={{min(component_values):.2f}}, max={{max(component_values):.2f}}, mean={{np.mean(component_values):.2f}}")
        
    def estimate_parameters(self):
        """Estimate transformation parameters using known prefix"""
        known_chunks = ["ict", "f{0"]  # ictf{ split into 3-char chunks with padding
        
        best_params = None
        min_global_error = float('inf')
        
        print("\nParameter estimation using known prefix 'ictf{'...")
        print("Searching parameter space...")
        
        for trial in range(10000):
            if trial % 1000 == 0:
                print(f"Trial {{trial}}/10000, best error so far: {{min_global_error:.4f}}")
                
            # Generate random transformation parameters
            r = np.random.randn(4)
            r /= np.linalg.norm(r)  # Unit quaternion
            
            # Random 3D rotation matrix (proper rotation)
            a = np.linalg.qr(np.random.randn(3, 3))[0]
            if np.linalg.det(a) < 0:
                a[:, 0] = -a[:, 0]
            
            s = np.random.uniform(0.8, 2.0)  # Scale factor
            t = np.random.uniform(-200, 200, size=3)  # Translation
            
            total_error = 0
            valid = True
            
            # Test on first two known chunks
            for i, chunk in enumerate(known_chunks):
                if i >= len(self.ar):
                    break;
                    
                ascii_vals = [ord(c) for c in chunk]
                
                # Try different w values for this chunk
                best_chunk_error = float('inf')
                for w_trial in range(50):
                    w = np.random.uniform(0, 255)  # Random first component
                    
                    # Forward transformation
                    original_quat = np.array([w] + ascii_vals)
                    rotated_quat = self.rotate(original_quat, r)
                    
                    # 3D transformation
                    transformed_spatial = s * (a @ rotated_quat[1:]) + t
                    predicted = np.array([rotated_quat[0]] + list(transformed_spatial))
                    
                    # Compare with actual data
                    error = np.linalg.norm(predicted - self.ar[i])
                    best_chunk_error = min(best_chunk_error, error)
                
                total_error += best_chunk_error
                
                if best_chunk_error > 20:  # Too large error for this chunk
                    valid = False
                    break
            
            if valid and total_error < min_global_error:
                min_global_error = total_error
                best_params = {
                    'r': r, 'a': a, 's': s, 't': t,
                    'error': total_error
                }
        
        return best_params
    
    def decode_with_params(self, params):
        """Decode all chunks using estimated parameters"""
        if not params:
            print("No valid parameters found!")
            return None
            
        print(f"\nDecoding with parameters (error: {{params['error']:.2f}})...")
        
        r = params['r']
        a = params['a']
        s = params['s']
        t = params['t']
        
        decoded_chunks = []
        
        for i, data in enumerate(self.ar):
            print(f"\nDecoding chunk {{i}}:")
            print(f"Input data: {{data}}")
            
            # Try multiple approaches to find valid ASCII characters
            best_chunk = None
            best_score = float('inf')
            
            # Try different w values to find the one that gives valid ASCII
            for w_candidate in np.linspace(0, 255, 100):
                try:
                    # Reverse the 3D transformation
                    w_transformed = data[0]
                    spatial_transformed = data[1:]
                    
                    # Inverse 3D transformation
                    spatial_rotated = np.linalg.solve(a, (spatial_transformed - t) / s)
                    
                    # Reconstruct quaternion and apply inverse rotation
                    quat_rotated = np.array([w_candidate] + list(spatial_rotated))
                    original_quat = self.inverse_rotate(quat_rotated, r)
                    
                    # Extract characters and check validity
                    chars = []
                    score = 0
                    valid_chars = True
                    
                    for val in original_quat[1:]:
                        char_code = int(round(val))
                        if 32 <= char_code <= 126:  # Printable ASCII
                            chars.append(chr(char_code))
                            # Prefer lowercase letters, digits, common symbols
                            if char_code >= 97 and char_code <= 122:  # lowercase
                                score += 1
                            elif char_code >= 48 and char_code <= 57:  # digits
                                score += 1
                            elif char_code in [95, 123, 125]:  # _, {, }
                                score += 1
                        else:
                            valid_chars = False
                            break
                    
                    if valid_chars and score > best_score:
                        best_score = score
                        chunk = ''.join(chars).rstrip('0')
                        best_chunk = chunk
                        
                except (np.linalg.LinAlgError, ValueError):
                    continue
            
            if best_chunk is not None:
                decoded_chunks.append(best_chunk)
                print(f"Decoded: '{{best_chunk}}'")
            else:
                decoded_chunks.append("???")
                print("Failed to decode valid characters")
        
        flag = ''.join(decoded_chunks)
        return flag
    
    def brute_force_analysis(self):
        """Analyze patterns without hardcoded flags"""
        print("\nBrute force analysis approach...")
        
        # Character frequency analysis from common CTF themes
        common_patterns = {
            'math_terms': ['quaternion', 'rotation', 'matrix', 'algebra', 'complex'],
            'crypto_terms': ['cipher', 'twisted', 'transform', 'encode', 'decode'],
            'ctf_terms': ['challenge', 'pwn', 'crypto', 'math', 'puzzle'],
            'connectors': ['_', 'is', 'and', 'the', 'of']
        }
        
        # Calculate expected middle length
        middle_length = len(self.ar) * 3 - 6  # Total chars - "ictf{}"
        print(f"Need {{middle_length}} characters between ictf{{ and }}")
        
        # Suggest analysis approach without giving away the answer
        print("Suggested approach:")
        print("1. Use mathematical optimization to find transformation parameters")
        print("2. Apply inverse transformations to recover ASCII values")
        print("3. Validate results against known prefix constraints")
        print("4. Look for common CTF/mathematical terminology patterns")
        
    def solve(self):
        """Main solving function"""
        self.analyze_structure()
        
        # Estimate parameters using known prefix
        params = self.estimate_parameters()
        
        if params:
            print(f"\nBest parameters found with error: {{params['error']:.2f}}")
            flag = self.decode_with_params(params)
            
            if flag and flag.startswith('ict'):
                print(f"\nDecoded flag: {{flag}}")
                return flag
        
        # Fallback to analysis approach
        self.brute_force_analysis()
        
        print("\nSolution requires implementing the full inverse transformation.")
        print("Key insight: Use the known 'ictf{' prefix to constrain parameter search.")
        return None

# Data from the challenge
data_points = [
    np.array([  17.33884894,   81.37080239, -143.96234736,  123.95164171]),
    np.array([ 168.34743674,  100.91788802, -135.90959582,  146.37617105]),
    np.array([ 157.94860314,   49.20197906, -155.2459834 ,   73.56498047]),
    np.array([   9.1131532 ,   49.36829422, -117.25335109,  181.11592151]),
    np.array([ 223.96684757,  -12.0765699 , -126.07584525,  125.88335439]),
    np.array([ 80.13452478,   40.78304285, -51.15180044, 143.18760932]),
    np.array([ 251.41332497,   48.04296984, -128.92087521,   68.4732401]),
    np.array([108.94539496,  -0.41865393, -53.94228136, 100.98194223]),
    np.array([183.06845007,  27.56200727, -52.57316992,  44.05723383]),
    np.array([ 96.56452698,  60.67582903, -76.44584757,  40.88253203])
]

if __name__ == "__main__":
    solver = QuaternionCTFSolver(data_points)
    result = solver.solve()