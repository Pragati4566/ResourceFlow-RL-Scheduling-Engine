import torch
from .workload_batch import BatchWorkload


class TensorizedBatchEnv:
    def __init__(self, num_envs=256, scenario='normal_load', max_queue=32,
                 max_active=4, max_steps=500, device='cpu'):
        self.num_envs = num_envs
        self.max_queue = max_queue
        self.max_active = max_active
        self.max_steps = max_steps
        self.device = device

        self.workload = BatchWorkload(scenario, num_envs, base_seed=42)
        config = self.workload.config

        self.vram_total_mb = config['vram_total_mb']
        self.model_base_memory_mb = config['model_base_memory_mb']
        self.kv_mb_per_token = config['kv_mb_per_token']
        self.max_concurrent = config['max_concurrent']
        self.prefill_speed = config['prefill_speed']
        self.decode_speed = config['decode_speed']
        self.memory_pressure_threshold = config.get('memory_pressure_threshold', 0.90)
        self.starvation_threshold = config.get('starvation_threshold', 100)
        self.memory_protect_threshold = config.get('memory_protect_threshold', 0.25)

        self._create_tensors()

    def _create_tensors(self):
        device = self.device
        n, q, a = self.num_envs, self.max_queue, self.max_active

        self.queue_prompt_len = torch.zeros(n, q, dtype=torch.int32, device=device)
        self.queue_output_len = torch.zeros(n, q, dtype=torch.int32, device=device)
        self.queue_priority = torch.zeros(n, q, dtype=torch.int32, device=device)
        self.queue_arrival_time = torch.zeros(n, q, dtype=torch.int32, device=device)
        self.queue_ttft_slo = torch.zeros(n, q, dtype=torch.int32, device=device)
        self.queue_deadline = torch.zeros(n, q, dtype=torch.int32, device=device)
        self.queue_valid = torch.zeros(n, q, dtype=torch.bool, device=device)
        self.queue_defer_count = torch.zeros(n, q, dtype=torch.int32, device=device)

        self.active_prompt_len = torch.zeros(n, a, dtype=torch.int32, device=device)
        self.active_output_len = torch.zeros(n, a, dtype=torch.int32, device=device)
        self.active_priority = torch.zeros(n, a, dtype=torch.int32, device=device)
        self.active_arrival_time = torch.zeros(n, a, dtype=torch.int32, device=device)
        self.active_ttft_slo = torch.zeros(n, a, dtype=torch.int32, device=device)
        self.active_deadline = torch.zeros(n, a, dtype=torch.int32, device=device)
        self.active_prefill_remaining = torch.zeros(n, a, dtype=torch.float32, device=device)
        self.active_decode_remaining = torch.zeros(n, a, dtype=torch.float32, device=device)
        self.active_generated_tokens = torch.zeros(n, a, dtype=torch.int32, device=device)
        self.active_first_token_time = torch.full((n, a), -1, dtype=torch.int32, device=device)
        self.active_valid = torch.zeros(n, a, dtype=torch.bool, device=device)

        self.env_time = torch.zeros(n, dtype=torch.int32, device=device)
        self.env_completions = torch.zeros(n, dtype=torch.int32, device=device)
        self.env_drops = torch.zeros(n, dtype=torch.int32, device=device)
        self.env_oom_events = torch.zeros(n, dtype=torch.int32, device=device)
        self.env_memory_pressure_events = torch.zeros(n, dtype=torch.int32, device=device)
        self.env_ttft_slo_miss_count = torch.zeros(n, dtype=torch.int32, device=device)
        self.env_deadline_miss_count = torch.zeros(n, dtype=torch.int32, device=device)
        self.env_tokens_generated = torch.zeros(n, dtype=torch.int32, device=device)
        self.env_gpu_idle_ticks = torch.zeros(n, dtype=torch.int32, device=device)
        self.env_next_job_id = torch.zeros(n, dtype=torch.int32, device=device)

        self.completed_details = []

        self.tick_completions = torch.zeros(n, dtype=torch.int32, device=device)
        self.tick_high_prio = torch.zeros(n, dtype=torch.int32, device=device)
        self.tick_drops = torch.zeros(n, dtype=torch.int32, device=device)
        self.tick_high_prio_drops = torch.zeros(n, dtype=torch.int32, device=device)
        self.tick_ttft_misses = torch.zeros(n, dtype=torch.int32, device=device)
        self.tick_deadline_misses = torch.zeros(n, dtype=torch.int32, device=device)
        self.tick_tokens = torch.zeros(n, dtype=torch.int32, device=device)
        self.tick_oom = torch.zeros(n, dtype=torch.int32, device=device)

    def set_scenario(self, scenario_name: str):
        self.workload.set_scenario(scenario_name)
        config = self.workload.config
        self.vram_total_mb = config['vram_total_mb']
        self.model_base_memory_mb = config['model_base_memory_mb']
        self.kv_mb_per_token = config['kv_mb_per_token']
        self.max_concurrent = config['max_concurrent']
        self.prefill_speed = config['prefill_speed']
        self.decode_speed = config['decode_speed']
        self.memory_pressure_threshold = config.get('memory_pressure_threshold', 0.90)
        self.starvation_threshold = config.get('starvation_threshold', 100)
        self.memory_protect_threshold = config.get('memory_protect_threshold', 0.25)

    def reset(self, seed=None):
        self.completed_details.clear()
        self.queue_valid.zero_()
        self.queue_defer_count.zero_()
        self.active_valid.zero_()
        self.active_first_token_time.fill_(-1)
        self.env_time.zero_()
        self.env_completions.zero_()
        self.env_drops.zero_()
        self.env_oom_events.zero_()
        self.env_memory_pressure_events.zero_()
        self.env_ttft_slo_miss_count.zero_()
        self.env_deadline_miss_count.zero_()
        self.env_tokens_generated.zero_()
        self.env_gpu_idle_ticks.zero_()
        self.env_next_job_id.zero_()
        self.queue_prompt_len.zero_()
        self.queue_output_len.zero_()
        self.queue_priority.zero_()
        self.queue_arrival_time.zero_()
        self.queue_ttft_slo.zero_()
        self.queue_deadline.zero_()
        self.active_prompt_len.zero_()
        self.active_output_len.zero_()
        self.active_priority.zero_()
        self.active_arrival_time.zero_()
        self.active_ttft_slo.zero_()
        self.active_deadline.zero_()
        self.active_prefill_remaining.zero_()
        self.active_decode_remaining.zero_()
        self.active_generated_tokens.zero_()

        if seed is not None:
            self.workload.reset(seed)
        else:
            self.workload.reset()

        return self._compute_observation()

    def step(self, actions):
        self.tick_completions.zero_()
        self.tick_high_prio.zero_()
        self.tick_drops.zero_()
        self.tick_high_prio_drops.zero_()
        self.tick_ttft_misses.zero_()
        self.tick_deadline_misses.zero_()
        self.tick_tokens.zero_()
        self.tick_oom.zero_()

        arrivals = self.workload.generate_arrivals(self.env_time)
        if arrivals is not None:
            self._add_arrivals(arrivals)

        self._process_active_jobs()

        vram = self._compute_vram_used()
        self._detect_oom(vram)
        self._detect_memory_pressure(vram)

        self._apply_actions(actions)

        self._detect_gpu_idle()

        self.env_time += 1

        obs = self._compute_observation()
        rewards = self._compute_reward(vram)
        done = (self.env_time >= self.max_steps).all().item()

        info = {
            'completions': self.tick_completions,
            'drops': self.tick_drops,
            'oom': self.tick_oom,
            'ttft_slo_misses': self.tick_ttft_misses,
            'deadline_misses': self.tick_deadline_misses,
        }

        return obs, rewards, done, info

    # ---- Arrivals ----

    def _add_arrivals(self, arrivals):
        env_ids = torch.arange(self.num_envs, device=self.device)
        arrival_mask = arrivals['prompt_len'] >= 0
        dispatch_envs = env_ids[arrival_mask]

        for i in range(len(dispatch_envs)):
            env_id = dispatch_envs[i].item()
            free = (~self.queue_valid[env_id]).nonzero(as_tuple=True)[0]
            if len(free) > 0:
                slot = free[0].item()
                self.queue_prompt_len[env_id, slot] = arrivals['prompt_len'][env_id]
                self.queue_output_len[env_id, slot] = arrivals['output_len'][env_id]
                self.queue_priority[env_id, slot] = arrivals['priority'][env_id]
                self.queue_arrival_time[env_id, slot] = arrivals['arrival_time'][env_id]
                self.queue_ttft_slo[env_id, slot] = arrivals['ttft_slo_ticks'][env_id]
                self.queue_deadline[env_id, slot] = arrivals['deadline_ticks'][env_id]
                self.queue_valid[env_id, slot] = True
                self.queue_defer_count[env_id, slot] = 0

    # ---- Active job processing ----

    def _process_active_jobs(self):
        batch_sizes = self.active_valid.sum(dim=1)
        eff = torch.ones(self.num_envs, device=self.device)
        eff = torch.where(batch_sizes <= 1, 1.0,
                torch.where(batch_sizes <= 2, 1.65 / 2,
                torch.where(batch_sizes <= 4, 2.80 / 4, 4.80 / 8)))
        eff_2d = eff.unsqueeze(1)

        prefill_mask = self.active_valid & (self.active_first_token_time < 0)
        eff_full = eff_2d.expand(-1, self.max_active)
        self.active_prefill_remaining[prefill_mask] -= (self.prefill_speed * eff_full)[prefill_mask]

        prefill_done = prefill_mask & (self.active_prefill_remaining <= 0)
        ft_time = self.env_time.unsqueeze(1).expand(-1, self.max_active)
        self.active_first_token_time[prefill_done] = ft_time[prefill_done]

        ttft = self.active_first_token_time.float() - self.active_arrival_time.float()
        ttft_miss = prefill_done & (ttft > self.active_ttft_slo.float())
        miss_counts = ttft_miss.sum(dim=1)
        self.tick_ttft_misses += miss_counts.int()
        self.env_ttft_slo_miss_count += miss_counts.int()

        decode_mask = self.active_valid & (self.active_first_token_time >= 0)
        advance = self.decode_speed * eff_full
        remaining = self.active_decode_remaining[decode_mask]
        actual = torch.minimum(advance[decode_mask], remaining)

        self.active_decode_remaining[decode_mask] -= actual
        active_gen_float = self.active_generated_tokens.float()
        active_gen_float[decode_mask] += actual
        self.active_generated_tokens = active_gen_float.int()

        env_indices = decode_mask.nonzero(as_tuple=True)[0]
        tokens_float = actual.float()
        tick_tokens = torch.zeros(self.num_envs, device=self.device)
        if len(env_indices) > 0:
            tick_tokens = tick_tokens.index_add_(0, env_indices, tokens_float)
        self.tick_tokens += tick_tokens.int()
        self.env_tokens_generated += tick_tokens.int()

        decode_done = decode_mask & (self.active_decode_remaining <= 0)
        comp_counts = decode_done.sum(dim=1)
        self.tick_completions += comp_counts.int()
        self.env_completions += comp_counts.int()

        high_prio = decode_done & (self.active_priority <= 2)
        self.tick_high_prio += high_prio.sum(dim=1).int()

        latency = self.env_time.unsqueeze(1).float() - self.active_arrival_time.float()
        dl_miss = decode_done & (latency > self.active_deadline.float())
        dl_counts = dl_miss.sum(dim=1)
        self.tick_deadline_misses += dl_counts.int()
        self.env_deadline_miss_count += dl_counts.int()

        done_indices = decode_done.nonzero(as_tuple=False)
        for idx in range(done_indices.size(0)):
            e = done_indices[idx, 0].item()
            s = done_indices[idx, 1].item()
            self.completed_details.append({
                'env_id': e,
                'completion_time': self.env_time[e].item(),
                'arrival_time': self.active_arrival_time[e, s].item(),
                'priority': self.active_priority[e, s].item(),
                'prompt_len': self.active_prompt_len[e, s].item(),
                'output_len': self.active_output_len[e, s].item(),
                'ttft_slo': self.active_ttft_slo[e, s].item(),
                'deadline': self.active_deadline[e, s].item(),
                'first_token_time': self.active_first_token_time[e, s].item(),
                'generated_tokens': self.active_generated_tokens[e, s].item(),
            })

        self.active_valid[decode_done] = False

    # ---- VRAM ----

    def _compute_vram_used(self):
        kv = (self.active_prompt_len.float() + self.active_generated_tokens.float()) * self.kv_mb_per_token
        kv_sum = (kv * self.active_valid.float()).sum(dim=1)
        return self.model_base_memory_mb + kv_sum

    def _detect_oom(self, vram):
        oom = vram > self.vram_total_mb
        self.tick_oom += oom.int()
        self.env_oom_events += oom.int()

    def _detect_memory_pressure(self, vram):
        pressure = vram / self.vram_total_mb > self.memory_pressure_threshold
        self.env_memory_pressure_events += pressure.int()

    def _detect_gpu_idle(self):
        active = self.active_valid.any(dim=1)
        has_queue = self.queue_valid.any(dim=1)
        self.env_gpu_idle_ticks += ((~active) & has_queue).int()

    # ---- Action dispatch ----

    def _apply_actions(self, actions):
        for env_id in range(self.num_envs):
            action = actions[env_id].item() if isinstance(actions, torch.Tensor) else actions[env_id]
            at = self.queue_arrival_time[env_id].float()
            tiebreak = at / 100000.0

            if action == 0:
                self._dispatch_single(env_id, at)
            elif action == 1:
                self._dispatch_single(env_id, self.queue_priority[env_id].float() + tiebreak)
            elif action == 2:
                self._dispatch_single(env_id, self.queue_prompt_len[env_id].float() + tiebreak)
            elif action == 3:
                total = self.queue_prompt_len[env_id].float() + self.queue_output_len[env_id].float()
                self._dispatch_single(env_id, total + tiebreak)
            elif action == 4:
                self._dispatch_batch(env_id, 2)
            elif action == 5:
                self._dispatch_batch(env_id, 4)
            elif action == 6:
                self._dispatch_batch(env_id, 8)
            elif action == 7:
                if self.active_valid[env_id].sum() < self.max_concurrent - 1:
                    self._dispatch_single(env_id, self.queue_priority[env_id].float() + tiebreak)
            elif action == 8:
                pass
            elif action == 9:
                self._defer(env_id)
            elif action == 10:
                self._drop_expired(env_id)
            elif action == 11:
                self._memory_protect_dispatch(env_id)

    def _dispatch_single(self, env_id, sort_key):
        valid = self.queue_valid[env_id]
        if not valid.any():
            return
        defer_penalty = self.queue_defer_count[env_id].float() * 1e9
        sentinel = torch.where(valid, sort_key + defer_penalty, torch.tensor(float('inf'), device=self.device))
        sorted_indices = sentinel.argsort()
        for idx in sorted_indices:
            idx = idx.item()
            if not self.queue_valid[env_id, idx]:
                continue
            if self._can_run_job(env_id, idx):
                self._start_job(env_id, idx)
                return

    def _dispatch_batch(self, env_id, batch_size):
        for _ in range(batch_size):
            valid = self.queue_valid[env_id]
            if not valid.any():
                break
            defer_penalty = self.queue_defer_count[env_id].float() * 1e9
            sort_key = self.queue_arrival_time[env_id].float() + defer_penalty
            sentinel = torch.where(valid, sort_key,
                                   torch.tensor(float('inf'), device=self.device))
            sorted_indices = sentinel.argsort()
            dispatched = False
            for idx in sorted_indices:
                idx = idx.item()
                if not self.queue_valid[env_id, idx]:
                    continue
                if self._can_run_job(env_id, idx):
                    self._start_job(env_id, idx)
                    dispatched = True
                    break
            if not dispatched:
                break

    def _start_job(self, env_id, queue_idx):
        free = (~self.active_valid[env_id]).nonzero(as_tuple=True)[0]
        if len(free) == 0:
            return
        slot = free[0].item()

        self.active_prompt_len[env_id, slot] = self.queue_prompt_len[env_id, queue_idx]
        self.active_output_len[env_id, slot] = self.queue_output_len[env_id, queue_idx]
        self.active_priority[env_id, slot] = self.queue_priority[env_id, queue_idx]
        self.active_arrival_time[env_id, slot] = self.queue_arrival_time[env_id, queue_idx]
        self.active_ttft_slo[env_id, slot] = self.queue_ttft_slo[env_id, queue_idx]
        self.active_deadline[env_id, slot] = self.queue_deadline[env_id, queue_idx]
        self.active_prefill_remaining[env_id, slot] = self.queue_prompt_len[env_id, queue_idx].float()
        self.active_decode_remaining[env_id, slot] = self.queue_output_len[env_id, queue_idx].float()
        self.active_generated_tokens[env_id, slot] = 0
        self.active_first_token_time[env_id, slot] = -1
        self.active_valid[env_id, slot] = True

        self.queue_valid[env_id, queue_idx] = False
        self.queue_defer_count[env_id, queue_idx] = 0

    def _can_run_job(self, env_id, queue_idx):
        job_kv = self.queue_prompt_len[env_id, queue_idx].float() * self.kv_mb_per_token
        current_vram = self._compute_vram_used()
        vram_ok = current_vram[env_id] + job_kv <= self.vram_total_mb
        slot_ok = self.active_valid[env_id].sum() < self.max_concurrent
        return vram_ok & slot_ok

    def _defer(self, env_id):
        valid = self.queue_valid[env_id]
        if not valid.any():
            return
        sentinel = torch.where(valid, -self.queue_priority[env_id].float(),
                               torch.tensor(float('inf'), device=self.device))
        worst_idx = sentinel.argmin().item()
        self.queue_defer_count[env_id, worst_idx] += 1

    def _drop_expired(self, env_id):
        valid = self.queue_valid[env_id]
        if not valid.any():
            return
        wait = self.env_time[env_id].float() - self.queue_arrival_time[env_id].float()
        expired = valid & (wait > self.queue_ttft_slo[env_id].float()) & (self.queue_priority[env_id] >= 3)
        expired_indices = expired.nonzero(as_tuple=True)[0]
        if len(expired_indices) > 0:
            drop_count = min(len(expired_indices), 2)
            for j in range(drop_count):
                idx = expired_indices[j].item()
                is_high_prio = self.queue_priority[env_id, idx] <= 2
                self.queue_valid[env_id, idx] = False
                self.tick_drops[env_id] += 1
                self.env_drops[env_id] += 1
                if is_high_prio:
                    self.tick_high_prio_drops[env_id] += 1

    def _memory_protect_dispatch(self, env_id):
        valid = self.queue_valid[env_id]
        if not valid.any():
            return
        threshold = self.vram_total_mb * self.memory_protect_threshold
        job_kv = self.queue_prompt_len[env_id].float() * self.kv_mb_per_token
        protect = valid & (job_kv <= threshold)
        if not protect.any():
            return
        sentinel = torch.where(protect, self.queue_arrival_time[env_id].float(),
                               torch.tensor(float('inf'), device=self.device))
        best_idx = sentinel.argmin().item()
        if self._can_run_job(env_id, best_idx):
            self._start_job(env_id, best_idx)

    # ---- Observation ----

    def _compute_observation(self):
        q_valid = self.queue_valid
        a_valid = self.active_valid
        n = self.num_envs

        q_len = q_valid.sum(dim=1).float()
        active_count = a_valid.sum(dim=1).float()
        free_slots = self.max_concurrent - active_count

        vram = self._compute_vram_used()
        vram_ratio = vram / self.vram_total_mb
        vram_headroom = 1.0 - vram_ratio

        q_prompt = self.queue_prompt_len.float()
        q_output = self.queue_output_len.float()
        q_prio = self.queue_priority.float()
        q_wait = self.env_time.unsqueeze(1).float() - self.queue_arrival_time.float()
        q_ttft = self.queue_ttft_slo.float()
        q_deadline = self.queue_deadline.float()

        safe_div = q_len.clamp(min=1)

        avg_prompt = (q_prompt * q_valid).sum(dim=1) / safe_div
        max_prompt = torch.where(q_valid, q_prompt, torch.tensor(0.0, device=self.device)).max(dim=1).values
        avg_output = (q_output * q_valid).sum(dim=1) / safe_div
        q_prio_sum = (q_prio * q_valid).sum(dim=1)
        avg_prio = torch.where(q_len > 0, q_prio_sum / q_len, torch.full_like(q_len, 3.0))

        high_prio = ((q_prio <= 2) & q_valid).sum(dim=1).float()
        oldest_wait = torch.where(q_valid, q_wait, torch.tensor(-1e9, device=self.device)).max(dim=1).values.clamp(min=0)
        avg_wait = (q_wait * q_valid).sum(dim=1) / safe_div
        near_ttft = (((q_wait / q_ttft.clamp(min=1)) > 0.8) & q_valid).sum(dim=1).float()
        near_deadline = (((q_wait / q_deadline.clamp(min=1)) > 0.8) & q_valid).sum(dim=1).float()
        short_jobs = (((q_prompt + q_output) < 300) & q_valid).sum(dim=1).float()
        long_context = ((q_prompt > 500) & q_valid).sum(dim=1).float()

        gpu_active = (active_count > 0).float()
        tokens_gen = (self.tick_tokens.float() / 20.0).clamp(max=1.0)
        ttft_flag = (self.env_ttft_slo_miss_count > 0).float()
        deadline_flag = (self.env_deadline_miss_count > 0).float()
        mem_flag = (self.env_memory_pressure_events > 5).float()

        obs = torch.stack([
            (q_len / 30.0).clamp(max=1.0),
            (active_count / self.max_concurrent).clamp(max=1.0),
            (free_slots / self.max_concurrent).clamp(max=1.0),
            vram_ratio.clamp(max=1.0),
            vram_headroom.clamp(max=1.0),
            (avg_prompt / 1000.0).clamp(max=1.0),
            (max_prompt / 2000.0).clamp(max=1.0),
            (avg_output / 2000.0).clamp(max=1.0),
            (avg_prio / 5.0).clamp(max=1.0),
            (high_prio / 10.0).clamp(max=1.0),
            (oldest_wait / 300.0).clamp(max=1.0),
            (avg_wait / 300.0).clamp(max=1.0),
            (near_ttft / 10.0).clamp(max=1.0),
            (near_deadline / 10.0).clamp(max=1.0),
            (short_jobs / 10.0).clamp(max=1.0),
            (long_context / 10.0).clamp(max=1.0),
            gpu_active,
            tokens_gen,
            ttft_flag,
            deadline_flag,
            mem_flag,
        ], dim=1)

        return obs.float()

    # ---- Reward ----

    def _compute_reward(self, vram):
        reward = (self.tick_completions.float() * 2.0
                  + self.tick_high_prio.float() * 3.0
                  + self.tick_tokens.float() * 0.05
                  - 4.0 * self.tick_ttft_misses.float()
                  - 6.0 * self.tick_deadline_misses.float()
                  - 10.0 * self.tick_oom.float()
                  - 2.0 * self.tick_drops.float()
                  - 1.0 * self.tick_high_prio_drops.float())

        active_count = self.active_valid.sum(dim=1).float()
        has_queue = self.queue_valid.any(dim=1).float()
        idle_slots = (self.max_concurrent - active_count).clamp(min=0)
        reward -= 0.5 * idle_slots * has_queue

        vram_ratio = vram / self.vram_total_mb
        reward -= 2.0 * (vram_ratio - 0.90).clamp(min=0) * 10.0

        wait_time = self.env_time.unsqueeze(1).float() - self.queue_arrival_time.float()
        oldest = torch.where(self.queue_valid, wait_time, torch.tensor(-1e9, device=self.device))
        oldest = oldest.max(dim=1).values.clamp(min=0)
        reward -= 3.0 * (oldest > self.starvation_threshold).float()

        return reward.clamp(-10.0, 10.0)
