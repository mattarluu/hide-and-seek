"""
parallel_env.py
Wrapper de entornos paralelos con diagnóstico avanzado.
"""

import numpy as np
from multiprocessing import Process, Pipe
import sys
import os
import time

def worker(rank, remote, parent_remote, env_fn, log_dir):
    """Worker process."""
    parent_remote.close()
    
    # Suprimir output del entorno si es ruidoso (opcional)
    # sys.stdout = open(os.devnull, 'w')

    env = None
    try:
        env = env_fn()
        remote.send(('ready', rank))
        
        while True:
            cmd, data = remote.recv()
            
            if cmd == 'step':
                try:
                    obs, done, rewards = env.step(data)
                    if done:
                        obs = env.reset()
                    remote.send(('result', (obs, done, rewards)))
                except Exception as e:
                    remote.send(('error', f"Step error in worker {rank}: {str(e)}"))
                    break
            
            elif cmd == 'reset':
                try:
                    obs = env.reset()
                    remote.send(('result', obs))
                except Exception as e:
                    remote.send(('error', f"Reset error in worker {rank}: {str(e)}"))
                    break
            
            elif cmd == 'close':
                break
            elif cmd == 'get_env':
                remote.send(('result', env))
            else:
                pass
                
    except KeyboardInterrupt:
        pass
    except Exception as e:
        try:
            remote.send(('error', f"Worker {rank} crash: {str(e)}"))
        except:
            pass
    finally:
        if env and hasattr(env, 'close'):
            try:
                env.close()
            except:
                pass
        try:
            remote.close()
        except:
            pass

class ParallelEnv:
    def __init__(self, env_fns, log_dir='./logs'):
        self.n_envs = len(env_fns)
        self.waiting = False
        self.closed = False
        
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.n_envs)])
        self.processes = []
        
        print(f"Starting {self.n_envs} workers...")
        for i, (work_remote, remote, env_fn) in enumerate(zip(self.work_remotes, self.remotes, env_fns)):
            args = (i, work_remote, remote, env_fn, log_dir)
            process = Process(target=worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()
        
        # Esperar confirmación de inicio
        print(f"Waiting for workers to come online...")
        ready_count = 0
        for i, remote in enumerate(self.remotes):
            if remote.poll(timeout=30.0):
                msg = remote.recv()
                if msg[0] == 'ready':
                    ready_count += 1
                elif msg[0] == 'error':
                    self.close()
                    raise RuntimeError(f"Worker {i} error on start: {msg[1]}")
            else:
                self.close()
                raise TimeoutError(f"Worker {i} timed out on start")
        
        print(f"✓ All {self.n_envs} workers ready!")

    def step_async(self, actions):
        if self.waiting:
            raise RuntimeError("Already waiting for step results")
            
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True
    
    def step_wait(self, timeout=300.0):
        """Espera inteligente: monitoriza quién falta."""
        if not self.waiting:
            raise RuntimeError("Not waiting for step results")
        
        results = [None] * self.n_envs
        waiting_for = set(range(self.n_envs))
        start_time = time.time()
        last_print = start_time
        
        while len(waiting_for) > 0:
            # Chequear timeout global
            now = time.time()
            if now - start_time > timeout:
                self.close()
                raise TimeoutError(f"TIMEOUT CRÍTICO: Los workers {list(waiting_for)} están bloqueados.")
            
            # Imprimir advertencia si tarda más de 5 segundos
            if now - start_time > 5.0 and now - last_print > 5.0:
                print(f"\n[⚠️ Alerta] Esperando a workers: {list(waiting_for)}... (Posible bucle infinito en Env)")
                last_print = now
            
            # Revisar mensajes de workers pendientes
            for i in list(waiting_for):
                if self.remotes[i].poll(): 
                    try:
                        msg = self.remotes[i].recv()
                        if msg[0] == 'result':
                            results[i] = msg[1]
                            waiting_for.remove(i)
                        elif msg[0] == 'error':
                            self.close()
                            raise RuntimeError(f"Worker {i} reportó error: {msg[1]}")
                    except EOFError:
                        self.close()
                        raise RuntimeError(f"Worker {i} cerró la conexión inesperadamente.")
            
            # Pequeña pausa para no quemar la CPU
            time.sleep(0.005)
                
        self.waiting = False
        observations, dones, rewards = zip(*results)
        return list(observations), list(dones), list(rewards)
    
    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()
    
    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        
        results = []
        for i, remote in enumerate(self.remotes):
            # Timeout generoso para reset
            if remote.poll(timeout=60.0):
                msg = remote.recv()
                if msg[0] == 'result':
                    results.append(msg[1])
                elif msg[0] == 'error':
                    raise RuntimeError(f"Worker {i} reset error: {msg[1]}")
            else:
                raise TimeoutError(f"Worker {i} timeout in reset")
        return results
    
    def close(self):
        if self.closed: return
        for remote in self.remotes:
            try: remote.send(('close', None))
            except: pass
        for p in self.processes:
            p.join(timeout=1)
            if p.is_alive(): 
                try: p.terminate()
                except: pass
        self.closed = True
        
    def __len__(self):
        return self.n_envs
    
    def __del__(self):
        if not self.closed:
            self.close()

# --- ESTA ES LA FUNCIÓN QUE FALTABA ---
def make_parallel_env(env_fn, n_envs):
    """Crear entorno paralelo."""
    env_fns = [env_fn for _ in range(n_envs)]
    return ParallelEnv(env_fns)