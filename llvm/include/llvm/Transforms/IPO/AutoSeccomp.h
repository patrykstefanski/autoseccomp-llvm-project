//===-- AutoSeccomp.h - AutoSeccomp pass ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass replaces each call to autoseccomp.restrict() with a call to
// __autoseccomp_restrict() taking as an argument the set of system call numbers
// reachable from that autoseccomp.restrict(). The pass also deletes all
// autoseccomp.type() intrinsics, which are necessary to compute the set of
// reachable system calls.
//
// We can implement __autoseccomp_restrict() to generate and install seccomp-bpf
// filter rules from the set passed as the argument. Thus, the pass can be used
// to simplify securing applications with seccomp.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_AUTOSECCOMP_H
#define LLVM_TRANSFORMS_IPO_AUTOSECCOMP_H

#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"

namespace llvm {
class AutoSeccompPass : public PassInfoMixin<AutoSeccompPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};
} // namespace llvm

#endif // LLVM_TRANSFORMS_IPO_AUTOSECCOMP_H
