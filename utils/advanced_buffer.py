import numpy as np

class SumTree:
    """
    A binary tree data structure where each parent node is the sum
    of its child nodes. Leaves store the 'priority' for individual samples.

    tree:
        Index layout:
            - The leaves that hold the actual priorities start at index `capacity - 1`
            - Internal nodes (parents) are stored in the indices [0 ... capacity-2]
        Example for capacity=4:
            tree indices: [0, 1, 2, 3, 4, 5, 6]
                         /        \
                        1          2    (internal nodes)
                       /  \      /  \
                      3    4    5    6  (leaves store actual priorities)
    """
    def __init__(self, capacity):
        self.capacity = capacity  # Max number of leaves (transitions)
        self.tree = np.zeros(2 * capacity - 1)  # Internal tree array
        self.data = np.zeros(capacity, dtype=object)  # To store actual samples
        self.write = 0  # Current position to write new data
        self.n_entries = 0  # Current number of written elements

    def total_priority(self):
        """Returns the sum of all priorities (the root node)."""
        return self.tree[0]

    def add(self, priority, data):
        """
        Add a new sample with given priority.
        If the buffer is not full, it appends.
        If the buffer is full, it overwrites the oldest sample in a circular fashion.
        """
        # Index of the leaf
        idx = self.write + self.capacity - 1

        self.data[self.write] = data  # Store sample
        self.update(idx, priority)    # Update tree with new priority

        # Move to next index, overwriting oldest if full
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, priority):
        """
        Update the priority of the leaf node at index `idx` to `priority`.
        Then propagate the change up the tree.
        """
        change = priority - self.tree[idx]
        self.tree[idx] = priority

        # Propagate the change up
        while idx != 0:
            idx = (idx - 1) // 2  # Move to parent index
            self.tree[idx] += change

    def get(self, value):
        """
        Get leaf index, priority, and data by traversing the tree.
        `value` is between 0 and total_priority, and we use it to find
        the leaf which corresponds to that cumulative priority.
        """
        parent_idx = 0

        # While non-leaf
        while True:
            left_child = 2 * parent_idx + 1
            right_child = left_child + 1

            # If at leaf, stop
            if left_child >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                # Descend the tree
                if value <= self.tree[left_child]:
                    parent_idx = left_child
                else:
                    value -= self.tree[left_child]
                    parent_idx = right_child

        data_idx = leaf_idx - (self.capacity - 1)
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Prioritized Replay Buffer.

    Args:
        capacity (int): Max size of the buffer (and underlying SumTree).
        alpha (float): Exponent for converting TD errors to priorities.
                       priority ~ (td_error + eps) ^ alpha
        beta (float): Importance-sampling weight exponent, typically annealed from
                      something like 0.4 -> 1.0 over time.
        beta_increment (float): Amount by which beta is incremented after each sampling.
        eps (float): Small constant added to TD errors to avoid zero priority.
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=1e-6, eps=1e-5):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.eps = eps

        # Keep a running max priority to add new samples with max priority
        # so they can be sampled at least once
        self.max_priority = 1.0

    def add(self, data, priority=None):
        """
        Add a transition (data) to the buffer.
        If priority is not given, use current max priority.
        """
        if priority is None:
            priority = self.max_priority
        # Convert to alpha-scaled priority
        p = (priority + self.eps) ** self.alpha
        self.tree.add(p, data)
        self.max_priority = max(self.max_priority, priority)

    def sample(self, batch_size=64):
        """
        Sample a batch of size `batch_size`.
        Returns:
            idxs (list of int): Leaf indices in the tree.
            batch (list): The sampled transitions/data.
            is_weights (np.array): Importance-sampling weights of shape (batch_size,).
        """
        batch = []
        idxs = []
        segment = self.tree.total_priority() / batch_size
        priorities = []

        # Beta typically anneals towards 1 with each sample
        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        # Compute importance-sampling weights
        # P(j) = priority_j / sum_of_all_priorities
        # w(j) = (1 / (N * P(j)))^(beta)
        # Then normalize by max w(j)
        total_p = self.tree.total_priority()
        sampling_probs = np.array(priorities) / total_p
        is_weights = np.power(self.tree.n_entries * sampling_probs, -self.beta)
        is_weights /= is_weights.max()  # Normalize

        return idxs, batch, is_weights

    def update(self, idx, new_priority):
        """
        Update the priority for a specific leaf index `idx`.
        Usually called after computing a new TD error for the sampled transition.
        """
        self.max_priority = max(self.max_priority, new_priority)
        p = (new_priority + self.eps) ** self.alpha
        self.tree.update(idx, p)


if __name__ == "__main__":
    import random

    buffer_size = 10
    prb = PrioritizedReplayBuffer(capacity=buffer_size, alpha=0.6, beta=0.4, beta_increment=1e-3)

    # Add some dummy transitions
    for i in range(15):
        # For demonstration, let's make the "priority" be i+1
        dummy_transition = (f"state_{i}", f"action_{i}", f"reward_{i}", f"next_state_{i}", False)
        prb.add(data=dummy_transition, priority=i + 1)

    # Sample a batch
    batch_size = 5
    idxs, batch, is_weights = prb.sample(batch_size=batch_size)
    print("Sampled leaf indices:", idxs)
    print("Sampled data:", batch)
    print("IS weights:", is_weights)

    # Let's say we compute new priorities (TD errors) after learning
    new_priorities = [random.uniform(1, 10) for _ in range(batch_size)]

    # Update the priorities in the tree
    for leaf_idx, p in zip(idxs, new_priorities):
        prb.update(leaf_idx, p)

