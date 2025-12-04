/*
 * Copyright (c) 2024, LMDeploy Contributors.
 * Adapted from TensorRT-LLM's speculativeDecodingMode.h
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#pragma once

#include <cstdint>

namespace turbomind {

class SpeculativeDecodingMode {
    // Speculative decoding mode flags
public:
    static auto constexpr None() {
        return SpeculativeDecodingMode{kNone};
    }

    static auto constexpr DraftTarget() {
        return SpeculativeDecodingMode{kDraftTarget};
    }

    static auto constexpr Eagle() {
        return SpeculativeDecodingMode{kEagle};
    }

    static auto constexpr NGram() {
        return SpeculativeDecodingMode{kNGram};
    }

    [[nodiscard]] bool constexpr isNone() const {
        return anyBitSet(kNone);
    }

    [[nodiscard]] bool constexpr isDraftTarget() const {
        return anyBitSet(kDraftTarget);
    }

    [[nodiscard]] bool constexpr isEagle() const {
        return anyBitSet(kEagle);
    }

    [[nodiscard]] bool constexpr isNGram() const {
        return anyBitSet(kNGram);
    }

    // Critical for KV cache management
    [[nodiscard]] bool constexpr needsKVCacheRewind() const {
        return anyBitSet(kDraftTarget | kEagle);
    }

    // Whether this mode requires attention masks
    [[nodiscard]] bool constexpr requiresAttentionMask() const {
        return anyBitSet(kDraftTarget | kEagle);
    }

    // Whether this mode predicts draft tokens
    [[nodiscard]] bool constexpr predictsDraftTokens() const {
        return anyBitSet(kDraftTarget | kEagle | kNGram);
    }

    using UnderlyingType = std::uint8_t;

    bool operator==(SpeculativeDecodingMode const& other) const {
        return mState == other.mState;
    }

    explicit constexpr SpeculativeDecodingMode(UnderlyingType state)
        : mState(state) {
    }

private:
    // No speculative decoding is used
    static UnderlyingType constexpr kNone{1U << 0U};
    // Draft/Target model approach
    static UnderlyingType constexpr kDraftTarget{1U << 1U};
    // EAGLE algorithm
    static UnderlyingType constexpr kEagle{1U << 2U};
    // NGram (prompt lookup) approach
    static UnderlyingType constexpr kNGram{1U << 3U};

    [[nodiscard]] bool constexpr anyBitSet(UnderlyingType bits) const {
        return (mState & bits) != 0;
    }

    [[nodiscard]] bool constexpr allBitSet(UnderlyingType bits) const {
        return (mState & bits) == bits;
    }

    UnderlyingType mState{kNone};
};

// Static assertions for correctness
static_assert(SpeculativeDecodingMode::None().isNone());
static_assert(!SpeculativeDecodingMode::None().isDraftTarget());
static_assert(!SpeculativeDecodingMode::None().isEagle());
static_assert(!SpeculativeDecodingMode::None().isNGram());

static_assert(SpeculativeDecodingMode::DraftTarget().isDraftTarget());
static_assert(!SpeculativeDecodingMode::DraftTarget().isNone());
static_assert(SpeculativeDecodingMode::DraftTarget().needsKVCacheRewind());

static_assert(SpeculativeDecodingMode::Eagle().isEagle());
static_assert(!SpeculativeDecodingMode::Eagle().isNone());
static_assert(SpeculativeDecodingMode::Eagle().needsKVCacheRewind());

static_assert(SpeculativeDecodingMode::NGram().isNGram());
static_assert(!SpeculativeDecodingMode::NGram().isNone());
static_assert(!SpeculativeDecodingMode::NGram().needsKVCacheRewind());

} // namespace turbomind
