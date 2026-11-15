"""Pull the kinematic chain out of a URDF. Structure only — no physics engine."""
import xml.etree.ElementTree as ET
import numpy as np

def chain(path):
    r = ET.parse(path).getroot()
    out = []
    for j in r.findall('joint'):
        o = j.find('origin')
        xyz = np.array([float(v) for v in (o.get('xyz','0 0 0').split())]) if o is not None else np.zeros(3)
        rpy = np.array([float(v) for v in (o.get('rpy','0 0 0').split())]) if o is not None else np.zeros(3)
        a = j.find('axis'); lim = j.find('limit')
        out.append(dict(
            name=j.get('name'), type=j.get('type'),
            parent=j.find('parent').get('link'), child=j.find('child').get('link'),
            xyz=xyz, rpy=rpy,
            axis=np.array([float(v) for v in a.get('xyz').split()]) if a is not None else None,
            lower=float(lim.get('lower')) if lim is not None and lim.get('lower') else None,
            upper=float(lim.get('upper')) if lim is not None and lim.get('upper') else None,
        ))
    return out

if __name__ == '__main__':
    import sys
    c = chain(sys.argv[1] if len(sys.argv)>1 else 'so101.urdf')
    print(f"{'joint':22} {'type':9} {'offset xyz (m)':>26} {'|d| mm':>7} {'axis':>10} {'limits (rad)':>20}")
    for j in c:
        d = np.linalg.norm(j['xyz'])*1000
        ax = ''.join('xyz'[i] for i,v in enumerate(j['axis']) if abs(v)>0.5) if j['axis'] is not None else '-'
        sgn = '-' if j['axis'] is not None and j['axis'].sum()<0 else ''
        lim = f"[{j['lower']:+.3f},{j['upper']:+.3f}]" if j['lower'] is not None else '-'
        print(f"{j['name']:22} {j['type']:9} {str(np.round(j['xyz'],4)):>26} {d:7.1f} {sgn+ax:>10} {lim:>20}")
