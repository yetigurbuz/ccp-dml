# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Adam with learning rate multipliers for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import gen_training_ops
from tensorflow.python.util.tf_export import keras_export

from tensorflow.keras.optimizers import Adam


@keras_export('keras.optimizers.AdamLRM')
class AdamLRM(Adam):
    def __init__(self,
                 learning_rate=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-7,
                 amsgrad=False,
                 lr_multiplier={},
                 name='AdamLRM',
                 **kwargs):
        r"""Construct a new Adam optimizer with learning rate multipliers.
            If amsgrad = False:
              Initialization:
              $$m_0 := 0 \text{(Initialize initial 1st moment vector)}$$
              $$v_0 := 0 \text{(Initialize initial 2nd moment vector)}$$
              $$t := 0 \text{(Initialize timestep)}$$
              The update rule for `variable` with gradient `g` uses an optimization
              described at the end of section 2 of the paper:
              $$t := t + 1$$
              $$lr_t := \text{learning\_rate} * \sqrt{1 - beta_2^t} / (1 - beta_1^t)$$
              $$m_t := beta_1 * m_{t-1} + (1 - beta_1) * g$$
              $$v_t := beta_2 * v_{t-1} + (1 - beta_2) * g * g$$
              $$variable := variable - lr_t * m_t / (\sqrt{v_t} + \epsilon)$$
            If amsgrad = True:
              Initialization:
              $$m_0 := 0 \text{(Initialize initial 1st moment vector)}$$
              $$v_0 := 0 \text{(Initialize initial 2nd moment vector)}$$
              $$v_hat_0 := 0 \text{(Initialize initial 2nd moment vector)}$$
              $$t := 0 \text{(Initialize timestep)}$$
              The update rule for `variable` with gradient `g` uses an optimization
              described at the end of section 2 of the paper:
              $$t := t + 1$$
              $$lr_t := \text{learning\_rate} * \sqrt{1 - beta_2^t} / (1 - beta_1^t)$$
              $$m_t := beta_1 * m_{t-1} + (1 - beta_1) * g$$
              $$v_t := beta_2 * v_{t-1} + (1 - beta_2) * g * g$$
              $$v_hat_t := max(v_hat_{t-1}, v_t)$$
              $$variable := variable - lr_t * m_t / (\sqrt{v_hat_t} + \epsilon)$$
            The default value of 1e-7 for epsilon might not be a good default in
            general. For example, when training an Inception network on ImageNet a
            current good choice is 1.0 or 0.1. Note that since AdamOptimizer uses the
            formulation just before Section 2.1 of the Kingma and Ba paper rather than
            the formulation in Algorithm 1, the "epsilon" referred to here is "epsilon
            hat" in the paper.
            The sparse implementation of this algorithm (used when the gradient is an
            IndexedSlices object, typically because of `tf.gather` or an embedding
            lookup in the forward pass) does apply momentum to variable slices even if
            they were not used in the forward pass (meaning they have a gradient equal
            to zero). Momentum decay (beta1) is also applied to the entire momentum
            accumulator. This means that the sparse behavior is equivalent to the dense
            behavior (in contrast to some momentum implementations which ignore momentum
            unless a variable slice was actually used).
            Args:
              learning_rate: A Tensor or a floating point value.  The learning rate.
              beta_1: A float value or a constant float tensor. The exponential decay
                rate for the 1st moment estimates.
              beta_2: A float value or a constant float tensor. The exponential decay
                rate for the 2nd moment estimates.
              epsilon: A small constant for numerical stability. This epsilon is
                "epsilon hat" in the Kingma and Ba paper (in the formula just before
                Section 2.1), not the epsilon in Algorithm 1 of the paper.
              amsgrad: boolean. Whether to apply AMSGrad variant of this algorithm from
                the paper "On the Convergence of Adam and beyond".
              lr_multiplier: A dictionary with variable names as keys and learning rate
                multipliers as values. Learning rates of all variables which names
                start with each key are multiplied by these values.
              name: Optional name for the operations created when applying gradients.
                Defaults to "Adam".
              **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
                `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
                gradients by value, `decay` is included for backward compatibility to
                allow time inverse decay of learning rate. `lr` is included for backward
                compatibility, recommended to use `learning_rate` instead.
            @compatibility(eager)
            When eager execution is enabled, `learning_rate`, `beta_1`, `beta_2`,
            and `epsilon` can each be a callable that takes no arguments and
            returns the actual value to use. This can be useful for changing these
            values across different invocations of optimizer functions.
            @end_compatibility
            """
        super(AdamLRM, self).__init__(learning_rate=learning_rate,
                                      beta_1=beta_1,
                                      beta_2=beta_2,
                                      epsilon=epsilon,
                                      amsgrad=amsgrad,
                                      name=name,
                                      **kwargs)

        self._lrm_names = list(lr_multiplier.keys())
        for k, v in lr_multiplier.items():
            self._set_hyper(f'lrm_{k}', v)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))

        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')

        # learning rate multipliers
        lr_t = coefficients['lr_t']
        for k in self._lrm_names:
            if var.name.startswith(k):
                lr_t = coefficients['lr_t'] * self._get_hyper(f'lrm_{k}', var.dtype)

        if not self.amsgrad:
            return gen_training_ops.ResourceApplyAdam(
                var=var.handle,
                m=m.handle,
                v=v.handle,
                beta1_power=coefficients['beta_1_power'],
                beta2_power=coefficients['beta_2_power'],
                lr=lr_t,
                beta1=coefficients['beta_1_t'],
                beta2=coefficients['beta_2_t'],
                epsilon=coefficients['epsilon'],
                grad=grad,
                use_locking=self._use_locking)
        else:
            vhat = self.get_slot(var, 'vhat')
            return gen_training_ops.ResourceApplyAdamWithAmsgrad(
                var=var.handle,
                m=m.handle,
                v=v.handle,
                vhat=vhat.handle,
                beta1_power=coefficients['beta_1_power'],
                beta2_power=coefficients['beta_2_power'],
                lr=lr_t,
                beta1=coefficients['beta_1_t'],
                beta2=coefficients['beta_2_t'],
                epsilon=coefficients['epsilon'],
                grad=grad,
                use_locking=self._use_locking)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, 'm')
        m_scaled_g_values = grad * coefficients['one_minus_beta_1_t']
        m_t = state_ops.assign(m, m * coefficients['beta_1_t'],
                               use_locking=self._use_locking)
        with ops.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, 'v')
        v_scaled_g_values = (grad * grad) * coefficients['one_minus_beta_2_t']
        v_t = state_ops.assign(v, v * coefficients['beta_2_t'],
                               use_locking=self._use_locking)
        with ops.control_dependencies([v_t]):
            v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)

        # learning rate multipliers
        lr = coefficients['lr']
        for k in self._lrm_names:
            if var.name.startswith(k):
                lr = coefficients['lr'] * self._get_hyper(f'lrm_{k}', var.dtype)

        if not self.amsgrad:
            v_sqrt = math_ops.sqrt(v_t)
            var_update = state_ops.assign_sub(
                var, lr * m_t / (v_sqrt + coefficients['epsilon']),
                use_locking=self._use_locking)
            return control_flow_ops.group(*[var_update, m_t, v_t])
        else:
            v_hat = self.get_slot(var, 'vhat')
            v_hat_t = math_ops.maximum(v_hat, v_t)
            with ops.control_dependencies([v_hat_t]):
                v_hat_t = state_ops.assign(
                    v_hat, v_hat_t, use_locking=self._use_locking)
            v_hat_sqrt = math_ops.sqrt(v_hat_t)
            var_update = state_ops.assign_sub(
                var,
                lr * m_t / (v_hat_sqrt + coefficients['epsilon']),
                use_locking=self._use_locking)
            return control_flow_ops.group(*[var_update, m_t, v_t, v_hat_t])
