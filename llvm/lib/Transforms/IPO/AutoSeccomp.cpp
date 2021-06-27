//===-- AutoSeccomp.cpp - AutoSeccomp pass ----------------------*- C++ -*-===//
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
// This pass will roughly do the following steps:
//
// 1. Split each basic block at the call instructions into multiple basic
//    blocks. This allows associating only one set of reachable system calls to
//    each basic block. Moreover, after this step, each call instruction is
//    followed by an unconditional branch, which makes the implementation
//    simpler.
//
// 2. Resolve all indirect call sites. The knowledge where the control might
//    flow for each program point is necessary to compute the set of reachable
//    system calls for that point. However, LLVM typically does not provide
//    information about possible targets for an indirect call. We use a
//    heuristic based on function signatures and taken addresses to
//    over-approximate the set of possible targets for each indirect call site.
//
//    When AutoSeccomp is enabled, Clang should:
//    - Annotate each function with the !type metadata, which provides the
//    information about the C type of the function.
//    - Prepend each indirect call site with a call to autoseccomp.type()
//    intrinsic. The intrinsic takes the function pointer of the indirect call,
//    and a metadata representing the C type of the function pointer.
//
//    We match each indirect call site to the functions using the following
//    rules:
//    - The type of the call site and the function are the same.
//    - The address of the function is taken, as only such functions can be
//    assigned to a function pointer.
//
//    In addition, we use the !callees metadata, if available for an indirect
//    call.
//
// 3. Build a program graph, where we connect each basic block to its
//    successors, both intra- and inter-procedural. Thus, the graph represents
//    the control-flow of the program. During this step, we also perform the
//    local analysis to find out which system call numbers each basic block
//    might invoke.
//
// 4. Coalesce strongly connected components (SCCs) in the program graph in
//    order to make the next step faster. The problem of finding reachable
//    system calls can be expressed as a data-flow analysis problem, where the
//    lattice is a set of system call numbers, and the analysis is a
//    backward-may analysis. However, each basic block within the same SCC will
//    have the same solution, and thus the basic blocks can be coalesced. This
//    step makes the program graph a directed acyclic graph.
//
//    We use Tarjan's algorithm to find SCCs, which in addition identifies SCCs
//    in a reverse topological sort order.
//
// 5. Propagate the system call numbers backwards through the program graph,
//    obtaining in the result a set of reachable system calls for each basic
//    block. Because of the previous step, the program graph is a directed
//    acyclic graph. Thus, we can visit each node only once to propagate the
//    system calls, without a need for an iterative algorithm that is typically
//    used to solve data-flow analysis problems.
//
// 6. Replace each autoseccomp.restrict() with a call to
//    __autoseccomp_restrict() passing as the argument the calculated set of
//    reachable system call numbers.
//
// 7. Remove all calls to the autoseccomp.type() intrinsic, which were added by
//    Clang.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/AutoSeccomp.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <string>
#include <system_error>

#define AUTOSECCOMP_DEBUG 1
#define OPTIMIZE_PTHREAD_CREATE 1

#if AUTOSECCOMP_DEBUG
#define DEBUG(x) (x)
#else
#define DEBUG(x)
#endif

using namespace llvm;

namespace {

cl::opt<bool> Disable("autoseccomp-disable", cl::desc("Disable AutoSeccomp"));

cl::opt<std::string>
    CallGraphFilename("autoseccomp-call-graph",
                      cl::desc("Dump call graph to specified file"),
                      cl::value_desc("filename"));

cl::opt<std::string>
    ProgramGraphFilename("autoseccomp-program-graph",
                         cl::desc("Dump program graph to specified file"),
                         cl::value_desc("filename"));

cl::opt<std::string> CoalescedProgramGraphFilename(
    "autoseccomp-coalesced-program-graph",
    cl::desc("Dump coalesced program graph to specified file"),
    cl::value_desc("filename"));

cl::opt<std::string> PropagatedProgramGraphFilename(
    "autoseccomp-propagated-program-graph",
    cl::desc("Dump propagated program graph to specified file"),
    cl::value_desc("filename"));

struct AutoSeccomp : public ModulePass {
  static char ID;

  AutoSeccomp() : ModulePass(ID) {
    initializeAutoSeccompPass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override;
};

} // namespace

INITIALIZE_PASS_BEGIN(AutoSeccomp, "auto-seccomp", "AutoSeccomp", false, false)
INITIALIZE_PASS_END(AutoSeccomp, "auto-seccomp", "AutoSeccomp", false, false)
char AutoSeccomp::ID = 0;

ModulePass *llvm::createAutoSeccompPass() { return new AutoSeccomp; }

PreservedAnalyses AutoSeccompPass::run(Module &M, ModuleAnalysisManager &AM) {
  AutoSeccomp Impl;
  bool Changed = Impl.runOnModule(M);
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

namespace {

struct SyscallNumberSet {
  static constexpr unsigned MaxNumber = 1024;

  BitVector BV;

  SyscallNumberSet() : BV(MaxNumber) {}

  SyscallNumberSet(const SyscallNumberSet &) = default;
  SyscallNumberSet &operator=(const SyscallNumberSet &) = default;

  SyscallNumberSet(SyscallNumberSet &&) = default;
  SyscallNumberSet &operator=(SyscallNumberSet &&) = default;

  void addNumber(unsigned Number) {
    assert(Number < MaxNumber);
    BV.set(Number);
  }

  void addNumbers(const BitVector &Numbers) {
    assert(Numbers.size() == MaxNumber);
    BV |= Numbers;
  }

  void join(const SyscallNumberSet &Other) { addNumbers(Other.BV); }

  bool contains(unsigned Number) const { return BV.test(Number); }

  bool empty() const { return BV.none(); }
};

raw_ostream &operator<<(raw_ostream &OS, const SyscallNumberSet &Set) {
  OS << '{';
  bool First = true;
  for (unsigned Nr = 0; Nr < SyscallNumberSet::MaxNumber; ++Nr) {
    if (Set.contains(Nr)) {
      if (First)
        First = false;
      else
        OS << ',';
      OS << Nr;
    }
  }
  OS << '}';
  return OS;
}

struct CallGraphNode {
  const Function &F;

  SmallVector<CallGraphNode *, 2> Predecessors;
  SmallVector<CallGraphNode *, 2> Successors;

  // For each call instruction within the function F we store a list of possible
  // target functions.
  using Targets = SmallVectorImpl<const Function *>;
  DenseMap<const CallBase *, const Targets *> CallTargets;

  // True if the function F or its callees might invoke autoseccomp.restrict().
  bool CallsAutoSeccompRestrict = false;

  explicit CallGraphNode(const Function &F) : F(F) {}

  CallGraphNode(const CallGraphNode &) = delete;
  void operator=(const CallGraphNode &) = delete;
};

struct ProgramGraphNode {
  const BasicBlock &BB;

  // A unique ID.
  unsigned ID;

  SmallVector<ProgramGraphNode *, 2> Predecessors{};
  SmallVector<ProgramGraphNode *, 2> Successors{};

  // The set of reachable syscall numbers from the beginning of the basic block
  // BB.
  SyscallNumberSet Gen;

  // Needed for Tarjan's SCCs algorithm.
  unsigned Index = 0;
  unsigned LowLink = 0;
  bool OnStack = false;
  ProgramGraphNode *Root = nullptr;

  explicit ProgramGraphNode(const BasicBlock &BB, unsigned ID)
      : BB(BB), ID(ID) {}

  ProgramGraphNode(const ProgramGraphNode &) = delete;
  void operator=(const ProgramGraphNode &) = delete;

  static void addEdge(ProgramGraphNode *From, ProgramGraphNode *To) {
    assert(From);
    assert(To);
    From->Successors.push_back(To);
    To->Predecessors.push_back(From);
  }

  static bool isEdgeInterprocedural(const ProgramGraphNode *From,
                                    const ProgramGraphNode *To) {
    assert(From);
    assert(To);
    return From->BB.getParent() != To->BB.getParent();
  }
};

struct AutoSeccompModule {
  Module &M;

  // Function type -> functions of that type.
  StringMap<SmallVector<const Function *, 1>> IndirectTargets{};

  DenseMap<const Function *, CallGraphNode *> CallGraphNodes;

  DenseMap<const BasicBlock *, ProgramGraphNode *> ProgramGraphNodes;
  unsigned ProgramGraphNodeID = 0;

  const Function *MainFunction = nullptr;
  const Function *ExitFunction = nullptr;

  // Needed for Tarjan's SCCs algorithm.
  SmallVector<ProgramGraphNode *, 64> SCCStack;
  unsigned SCCIndex = 1;

  // An order in which we should propagate reachable syscall numbers.
  SmallVector<ProgramGraphNode *, 0> Postorder;

  explicit AutoSeccompModule(Module &M) : M(M) {}

  CallGraphNode *createCallGraphNode(const Function &F) {
    auto *Node = new CallGraphNode(F);
    [[maybe_unused]] auto [_, Inserted] = CallGraphNodes.insert({&F, Node});
    assert(Inserted);
    return Node;
  }

  CallGraphNode *getCallGraphNode(const Function &F) {
    auto It = CallGraphNodes.find(&F);
    return It != CallGraphNodes.end() ? It->second : nullptr;
  }

  const CallGraphNode *getCallGraphNode(const Function &F) const {
    return const_cast<AutoSeccompModule *>(this)->getCallGraphNode(F);
  }

  CallGraphNode *getOrCreateCallGraphNode(const Function &F) {
    if (auto *Node = getCallGraphNode(F))
      return Node;
    return createCallGraphNode(F);
  }

  ProgramGraphNode *createProgramGraphNode(const BasicBlock &BB) {
    auto *Node = new ProgramGraphNode(BB, ++ProgramGraphNodeID);
    [[maybe_unused]] auto [_, Inserted] = ProgramGraphNodes.insert({&BB, Node});
    assert(Inserted);
    return Node;
  }

  ProgramGraphNode *getProgramGraphNode(const BasicBlock &BB) {
    auto It = ProgramGraphNodes.find(&BB);
    return It != ProgramGraphNodes.end() ? It->second : nullptr;
  }

  const ProgramGraphNode *getProgramGraphNode(const BasicBlock &BB) const {
    return const_cast<AutoSeccompModule *>(this)->getProgramGraphNode(BB);
  }

  ProgramGraphNode *getOrCreateProgramGraphNode(const BasicBlock &BB) {
    if (auto *Node = getProgramGraphNode(BB))
      return Node;
    return createProgramGraphNode(BB);
  }

  bool run();
  void splitBasicBlocks();
  void initIndirectTargets();
  void buildCallGraph();
  void propagateCallsAutoSeccompRestrict();
  void buildProgramGraph();
  void findSCCs();
  void strongConnect(ProgramGraphNode *Node);
  void coalesceSCCs();
  void propagateSyscallSet();
  void replaceRestrictIntrinsics();
  Value *createSyscallsBitVector(const ProgramGraphNode *Node, unsigned ID);
  void deleteTypeIntrinsics();
  void dumpCallGraph(StringRef Filename) const;
  void dumpProgramGraph(bool Coalesced, StringRef Filename) const;
};

bool AutoSeccomp::runOnModule(Module &M) { return AutoSeccompModule(M).run(); }

bool AutoSeccompModule::run() {
  if (Disable)
    return false;

  // Split each basic block at the call instructions into multiple basic blocks.
  DEBUG(errs() << "--------------- Split Basic Blocks ---------------\n");
  splitBasicBlocks();

  // Collect all functions whose address is taken, and their types.
  DEBUG(errs() << "------------- Init Indirect Targets --------------\n");
  initIndirectTargets();

#if AUTOSECCOMP_DEBUG
  for (const auto &Entry : IndirectTargets) {
    llvm::errs() << "  " << Entry.getKey() << ":\n";
    for (const llvm::Function *F : Entry.getValue()) {
      llvm::errs() << "    " << F->getName() << '\n';
    }
  }
#endif

  // Build a call graph, and resolve each interesting call instruction within
  // each function.
  DEBUG(errs() << "---------------- Build Call Graph ----------------\n");
  buildCallGraph();

  if (!CallGraphFilename.empty()) {
    dumpCallGraph(CallGraphFilename);
  }

  DEBUG(errs() << "------ Propagate 'CallsAutoSeccompRestrict' ------\n");
  propagateCallsAutoSeccompRestrict();

  // Build a program graph for the whole program. This also performs the local
  // analysis to find out which syscalls might be invoked.
  DEBUG(errs() << "-------------- Build Program Graph ---------------\n");
  buildProgramGraph();

  DEBUG(errs() << "------------------- Find SCCs --------------------\n");
  findSCCs();

  if (!ProgramGraphFilename.empty()) {
    dumpProgramGraph(/*Coalesced=*/false, ProgramGraphFilename);
  }

  DEBUG(errs() << "----------------- Coalesce SCCs ------------------\n");
  coalesceSCCs();

  if (!CoalescedProgramGraphFilename.empty()) {
    dumpProgramGraph(/*Coalesced=*/true, CoalescedProgramGraphFilename);
  }

  DEBUG(errs() << "------------- Propagate Syscall Set --------------\n");
  propagateSyscallSet();

  if (!PropagatedProgramGraphFilename.empty()) {
    dumpProgramGraph(/*Coalesced=*/true, PropagatedProgramGraphFilename);
  }

  DEBUG(errs() << "---------- Replace Restrict Intrinsics -----------\n");
  replaceRestrictIntrinsics();

  DEBUG(errs() << "------------- Delete Type Intrinsics -------------\n");
  deleteTypeIntrinsics();

  return true;
}

// We split each basic block after each call instruction occurring in the block.
// This makes each call instruction to be followed by an unconditional branch.
void AutoSeccompModule::splitBasicBlocks() {
  SmallVector<BasicBlock *, 128> Worklist;
  for (Function &F : M) {
    assert(Worklist.empty());
    Worklist.reserve(F.size());

    for (BasicBlock &BB : F)
      Worklist.push_back(&BB);

    while (!Worklist.empty()) {
      BasicBlock *BB = Worklist.pop_back_val();

      for (auto I = BB->begin(); I != BB->end(); ++I) {
        if (isa<CallBase>(*I)) {
          ++I;
          // Avoid splitting calls followed by an unconditional branch.
          if (const auto *Br = dyn_cast<BranchInst>(&*I);
              Br && Br->isUnconditional()) {
            break;
          }

          BasicBlock *NewBB = BB->splitBasicBlock(I);
          Worklist.push_back(NewBB);
          break;
        }
      }
    }
  }
}

// Makes the mapping: function type -> functions of that type.
void AutoSeccompModule::initIndirectTargets() {
  for (const Function &F : M) {
    if (F.hasAddressTaken()) {
      const MDNode *Node = F.getMetadata(LLVMContext::MD_type);
      if (!Node) {
        errs() << "AutoSeccomp: Unknown type for " << F.getName() << '\n';
        continue;
      }
      assert(Node->getNumOperands() == 2);
      const Metadata *MD = Node->getOperand(1).get();
      const MDString *Ty = cast<MDString>(MD);
      IndirectTargets[Ty->getString()].push_back(&F);
    }
  }
}

void AutoSeccompModule::buildCallGraph() {
  auto CreateTargetSingleton = [](const Function *Callee) {
    // TODO: We should delete the vector when the analysis exits.
    return new SmallVector<const Function *, 1>{Callee};
  };

  DenseMap<const Value *, const MDString *> ValueType;
  SmallVector<const Value *, 4> Worklist;
  for (const Function &F : M) {
    CallGraphNode *Node = getOrCreateCallGraphNode(F);
    ValueType.clear();

    // For each call to the autoseccomp.type() intrinsic, we mark with the type
    // the values that are used to derive the function pointer passed to the
    // intrinsic.
    for (const BasicBlock &BB : F) {
      auto I = ++BB.rbegin();
      if (I == BB.rend())
        continue;

      const auto *Int = dyn_cast<IntrinsicInst>(&*I);
      if (!Int || Int->getIntrinsicID() != Intrinsic::autoseccomp_type)
        continue;

      const MDString *TypeMD = cast<MDString>(
          cast<MetadataAsValue>(Int->getOperand(1))->getMetadata());
      assert(Worklist.empty());
      Worklist.push_back(Int->getOperand(0));
      while (!Worklist.empty()) {
        const Value *V = Worklist.pop_back_val();
        ValueType[V] = TypeMD;
        if (isa<BitCastInst>(V) || isa<IntToPtrInst>(V)) {
          const Value *Op = cast<CastInst>(V)->getOperand(0);
          if (ValueType.find(Op) == ValueType.end())
            Worklist.push_back(Op);
        } else if (const auto *PHI = dyn_cast<PHINode>(V)) {
          for (const Value *Op : PHI->incoming_values()) {
            if (ValueType.find(Op) == ValueType.end())
              Worklist.push_back(Op);
          }
        }
      }
    }

    // Assign a list of target functions to each interesting call instruction.
    for (const BasicBlock &BB : F) {
      auto I = ++BB.rbegin();
      if (I == BB.rend())
        continue;

      const auto *Call = dyn_cast<CallBase>(&*I);
      if (!Call)
        continue;

      if (const auto *Int = dyn_cast<IntrinsicInst>(Call);
          Int && Int->getIntrinsicID() == Intrinsic::autoseccomp_restrict) {
        Node->CallsAutoSeccompRestrict = true;
        continue;
      }

      if (const Function *Callee = Call->getCalledFunction()) {
        Node->CallTargets[Call] = CreateTargetSingleton(Callee);
        continue;
      }

      const Value *Operand = Call->getCalledOperand();

      if (isa<InlineAsm>(Operand))
        continue;

      // An alias to a function (e.g., weak alias to malloc()). Since we
      // assume static builds, we can resolve the aliases now.
      if (const auto *GA = dyn_cast<GlobalAlias>(Operand)) {
        if (const auto *Callee = dyn_cast<Function>(GA->getAliasee())) {
          Node->CallTargets[Call] = CreateTargetSingleton(Callee);
          continue;
        }
      }

      if (const auto *BitCast = dyn_cast<BitCastOperator>(Operand)) {
        const auto *Op = BitCast->getOperand(0);
        assert(!isa<BitCastOperator>(Op) && "Multiple bitcasts?");
        if (const auto *Callee = dyn_cast<Function>(Op)) {
          Node->CallTargets[Call] = CreateTargetSingleton(Callee);
          continue;
        }
        if (const auto *GA = dyn_cast<GlobalAlias>(Op)) {
          if (const auto *Callee = dyn_cast<Function>(GA->getAliasee())) {
            Node->CallTargets[Call] = CreateTargetSingleton(Callee);
            continue;
          }
        }
      }

      // The callees metadata provides a list of possible targets. If it is
      // available, we can use it instead of our heuristic.
      if (const MDNode *MD = Call->getMetadata(LLVMContext::MD_callees)) {
        // TODO: We should delete the vector when the analysis exits.
        auto *Targets = new SmallVector<const Function *, 4>;
        for (const auto &Op : MD->operands()) {
          const Function *Callee = mdconst::extract_or_null<Function>(Op);
          if (Callee)
            Targets->push_back(Callee);
        }
        Node->CallTargets[Call] = Targets;
        continue;
      }

      // Match targets by the function signature.
      const MDString *TypeMD = nullptr;
      assert(Worklist.empty());
      Worklist.push_back(Operand);
      while (!Worklist.empty()) {
        const Value *V = Worklist.pop_back_val();
        auto It = ValueType.find(V);
        if (It != ValueType.end()) {
          TypeMD = It->second;
          Worklist.clear();
          break;
        }
        if (isa<BitCastInst>(V) || isa<IntToPtrInst>(V)) {
          Worklist.push_back(cast<CastInst>(V)->getOperand(0));
        } else if (const auto *PHI = dyn_cast<PHINode>(V)) {
          for (const Value *V : PHI->incoming_values()) {
            Worklist.push_back(V);
          }
        }
      }
      if (!TypeMD) {
        errs() << "AutoSeccomp: Incorrect indirect call\n"
               << "Call: " << *Call << '\n'
               << "Operand: " << *Operand << '\n'
               << F;
        continue;
      }
      const auto &Targets = IndirectTargets[TypeMD->getString()];
      Node->CallTargets[Call] = &Targets;
    }
  }

  // Build the call graph.
  for (const Function &F : M) {
    CallGraphNode *Node = getCallGraphNode(F);
    assert(Node);

    // Unionize each set of targets into a set of successors.
    auto &Successors = Node->Successors;
    for (auto [_, Targets] : Node->CallTargets) {
      for (const Function *Target : *Targets) {
        CallGraphNode *TargetNode = getCallGraphNode(*Target);
        assert(TargetNode);
        Successors.push_back(TargetNode);
      }
    }

    // Remove possible duplicates.
    std::sort(Successors.begin(), Successors.end());
    const auto *Last = std::unique(Successors.begin(), Successors.end());
    Successors.erase(Last, Successors.end());

    for (CallGraphNode *Succ : Successors)
      Succ->Predecessors.push_back(Node);
  }
}

void AutoSeccompModule::propagateCallsAutoSeccompRestrict() {
  SmallVector<CallGraphNode *, 128> Worklist;

  for (const Function &F : M) {
    CallGraphNode *Node = getCallGraphNode(F);
    assert(Node);
    if (Node->CallsAutoSeccompRestrict)
      Worklist.push_back(Node);
  }

  while (!Worklist.empty()) {
    CallGraphNode *Node = Worklist.pop_back_val();
    for (CallGraphNode *Pred : Node->Predecessors) {
      if (!Pred->CallsAutoSeccompRestrict) {
        Pred->CallsAutoSeccompRestrict = true;
        Worklist.push_back(Pred);
      }
    }
  }
}

// Builds a program graph and performs the local analysis.
void AutoSeccompModule::buildProgramGraph() {
  DenseMap<const Function *, SmallVector<const BasicBlock *, 1>> FuncExits;

  for (const Function &F : M) {
    StringRef Name = F.getName();
    if (Name == "main") {
      assert(!MainFunction);
      MainFunction = &F;
    } else if (Name == "exit") {
      assert(!ExitFunction);
      ExitFunction = &F;
    }

    FuncExits[&F] = {};
    for (const BasicBlock &BB : F) {
      if (const auto *Return = dyn_cast<ReturnInst>(&BB.back())) {
        FuncExits[&F].push_back(&BB);
      }
    }
  }

  // Intraprocedural links.
  for (const Function &F : M) {
    for (const BasicBlock &BB : F) {
      auto *From = getOrCreateProgramGraphNode(BB);
      for (const BasicBlock *SuccBB : successors(&BB)) {
        auto *To = getOrCreateProgramGraphNode(*SuccBB);
        ProgramGraphNode::addEdge(From, To);
      }
    }
  }

  // Add a link from main() to exit().
  if (MainFunction && ExitFunction) {
    auto *ExitFEntry = getProgramGraphNode(ExitFunction->getEntryBlock());
    for (const BasicBlock *BB : FuncExits[MainFunction]) {
      auto *MainFExit = getProgramGraphNode(*BB);
      ProgramGraphNode::addEdge(MainFExit, ExitFEntry);
    }
  }

  SmallVector<ProgramGraphNode *, 64> RestrictNodes;

  // Interprocedural links.
  for (const Function &F : M) {
    const CallGraphNode *CGN = getCallGraphNode(F);
    assert(CGN);

    for (const BasicBlock &BB : F) {
      auto I = ++BB.rbegin();
      if (I == BB.rend())
        continue;

      const auto *Call = dyn_cast<CallBase>(&*I);
      if (!Call)
        continue;

      auto *CallNode = getProgramGraphNode(BB);

      // The BB should have a single successor, as we have split the blocks at
      // call sites.
      const BasicBlock *SuccBB = BB.getSingleSuccessor();
      assert(SuccBB);
      auto *ReturnNode = getProgramGraphNode(*SuccBB);

      if (const auto *Int = dyn_cast<IntrinsicInst>(Call)) {
        Intrinsic::ID ID = Int->getIntrinsicID();
        // TODO: We must handle non-leaf intrinsics as well.
        assert(Intrinsic::isLeaf(ID));
        if (ID == Intrinsic::autoseccomp_restrict)
          RestrictNodes.push_back(CallNode);
        continue;
      }

      auto HandleSyscall = [&](const CallBase *Call) {
        if (const auto *ConstInt =
                dyn_cast<ConstantInt>(Call->getArgOperand(0))) {
          uint64_t Nr = ConstInt->getZExtValue();
          if (Nr >= SyscallNumberSet::MaxNumber) {
            errs() << "AutoSeccomp: Syscall " << Nr << " too big\n";
          } else {
            CallNode->Gen.addNumber(Nr);
          }
        } else {
          StringRef FuncName = Call->getParent()->getParent()->getName();
          errs() << "AutoSeccomp: Syscall with non-constant number in "
                 << FuncName << ":\n"
                 << "  Call: " << *Call << '\n';
        }
      };

      if (const auto *Asm = dyn_cast<InlineAsm>(Call->getCalledOperand())) {
        const std::string &AsmStr = Asm->getAsmString();
        if (AsmStr == "syscall") {
          HandleSyscall(Call);
        } else {
          errs() << "AutoSeccomp: Unhandled inline assembly: '" << AsmStr
                 << "'\n";
        }
        continue;
      }

      auto It = CGN->CallTargets.find(Call);
      if (It == CGN->CallTargets.end()) {
        errs() << "AutoSeccomp: Unhandled call case:\n"
               << "  Call: " << *Call << '\n'
               << "  Called operand: " << *Call->getCalledOperand() << '\n';
        continue;
      }

      const auto *Targets = It->second;
      for (const Function *Callee : *Targets) {
        assert(!Callee->isIntrinsic());

        StringRef Name = Callee->getName();

        // __syscall and __syscall_cp are musl specific.
        if (Name == "syscall" || Name == "__syscall" ||
            Name == "__syscall_cp") {
          HandleSyscall(Call);
          continue;
        }

        // Optimize the call to pthread_create().
        // We add edges between the call node to the start routines passed to
        // the pthread_create() call. This improves the precision of the program
        // graph without a context-sensitive analysis.
        if (OPTIMIZE_PTHREAD_CREATE &&
            (Name == "pthread_create" || Name == "__pthread_create")) {
          // autoconf can make a call without arguments...
          if (Call->getNumArgOperands() == 4) {
            const Value *Op = Call->getArgOperand(2);
            if (const Function *StartRoutine = dyn_cast<Function>(Op)) {
              DEBUG(errs()
                    << "AutoSeccomp: Handling pthread_create(): Adding direct "
                       "edge to "
                    << StartRoutine->getName() << '\n');
              const BasicBlock &EntryBB = StartRoutine->getEntryBlock();
              auto *EntryNode = getProgramGraphNode(EntryBB);
              ProgramGraphNode::addEdge(CallNode, EntryNode);
            } else {
              auto It = IndirectTargets.find("_ZTSFPvS_E"); // void *(*)(void *)
              if (It != IndirectTargets.end()) {
                for (const Function *StartRoutine : It->second) {
                  DEBUG(errs()
                        << "AutoSeccomp: Handling pthread_create(): Adding "
                           "indirect edge to "
                        << StartRoutine->getName() << '\n');
                  const BasicBlock &EntryBB = StartRoutine->getEntryBlock();
                  auto *EntryNode = getProgramGraphNode(EntryBB);
                  ProgramGraphNode::addEdge(CallNode, EntryNode);
                }
              }
            }
          }
        }

        if (Callee->isDeclaration()) {
          // Special cases for musl's functions implemented in assembly.
          if (Name == "memcmp" || Name == "memcpy" || Name == "memset") {
            continue;
          }
          if (Name == "__restore_rt") {
            CallNode->Gen.addNumber(15); // SYS_rt_sigreturn
            continue;
          }
          if (Name == "__set_thread_area") {
            CallNode->Gen.addNumber(158); // SYS_prctl
            continue;
          }
          if (Name == "__unmapself") {
            CallNode->Gen.addNumber(11); // SYS_munmap
            CallNode->Gen.addNumber(60); // SYS_exit
            continue;
          }
          if (Name == "__clone") {
            CallNode->Gen.addNumber(56); // SYS_clone
            CallNode->Gen.addNumber(60); // SYS_exit

            // Optimize the call to __clone() within pthread_create().
            if (OPTIMIZE_PTHREAD_CREATE &&
                (F.getName() == "pthread_create" ||
                 F.getName() == "__pthread_create")) {
              DEBUG(errs() << "AutoSeccomp: Handling clone(): In "
                           << F.getName() << '\n');

              // pthread_create() invokes __clone() with start() or start_c11()
              // as the start routine. We handle both cases here, and ignore the
              // indirect call in the start() or start_c11() to the function
              // passed to pthread_create().

              // start() invokes the following system calls:
              CallNode->Gen.addNumber(14);  // SYS_rt_sigprocmask
              CallNode->Gen.addNumber(202); // __wait -> SYS_futex
              CallNode->Gen.addNumber(218); // SYS_set_tid_address

              // Both start() and start_c11() invoke pthread_exit().
              const Function *PthreadExit = M.getFunction("__pthread_exit");
              assert(PthreadExit);
              const BasicBlock &EntryBB = PthreadExit->getEntryBlock();
              auto *EntryNode = getProgramGraphNode(EntryBB);
              ProgramGraphNode::addEdge(CallNode, EntryNode);
              continue;
            }

            assert(Call->getNumArgOperands() >= 1);
            const Value *Op = Call->getArgOperand(0);
            if (const Function *StartRoutine = dyn_cast<Function>(Op)) {
              DEBUG(errs()
                    << "AutoSeccomp: Handling clone(): Adding direct edge to "
                    << StartRoutine->getName() << '\n');
              const BasicBlock &EntryBB = StartRoutine->getEntryBlock();
              auto *EntryNode = getProgramGraphNode(EntryBB);
              ProgramGraphNode::addEdge(CallNode, EntryNode);
            } else {
              auto It = IndirectTargets.find("_ZTSFiPvE"); // int(*)(void *)
              if (It != IndirectTargets.end()) {
                for (const Function *StartRoutine : It->second) {
                  DEBUG(errs()
                        << "AutoSeccomp: Handling clone(): Adding indirect "
                           "edge to "
                        << StartRoutine->getName() << '\n');
                  const BasicBlock &EntryBB = StartRoutine->getEntryBlock();
                  auto *EntryNode = getProgramGraphNode(EntryBB);
                  ProgramGraphNode::addEdge(CallNode, EntryNode);
                }
              }
            }

            continue;
          }

          DEBUG(errs() << "AutoSeccomp: Unhandled declaration: " << Name
                       << '\n');
          continue;
        }

        const BasicBlock &EntryBB = Callee->getEntryBlock();
        auto *EntryNode = getProgramGraphNode(EntryBB);
        ProgramGraphNode::addEdge(CallNode, EntryNode);

        if (Callee->hasFnAttribute(Attribute::NoReturn)) {
          DEBUG(errs() << "AutoSeccomp: NoReturn: " << Name << '\n');
          continue;
        }

        const CallGraphNode *CalleeCGN = getCallGraphNode(*Callee);
        if (CalleeCGN->CallsAutoSeccompRestrict) {
          for (const BasicBlock *ExitBB : FuncExits[Callee]) {
            auto *ExitNode = getProgramGraphNode(*ExitBB);
            ProgramGraphNode::addEdge(ExitNode, ReturnNode);
          }
        }
      }
    }
  }

  for (ProgramGraphNode *Node : RestrictNodes) {
    for (ProgramGraphNode *Pred : Node->Predecessors) {
      constexpr int SysPrctl = 157;
      Pred->Gen.addNumber(SysPrctl);
    }
  }
}

void AutoSeccompModule::findSCCs() {
  // Start with the main function to make exported program graphs prettier.
  if (MainFunction) {
    auto *InitNode = getProgramGraphNode(MainFunction->getEntryBlock());
    strongConnect(InitNode);
  }

  for (auto &[_, Node] : ProgramGraphNodes) {
    if (!Node->Root)
      strongConnect(Node);
  }
}

void AutoSeccompModule::strongConnect(ProgramGraphNode *Node) {
  Node->Index = SCCIndex;
  Node->LowLink = SCCIndex;
  ++SCCIndex;
  SCCStack.push_back(Node);
  Node->OnStack = true;

  for (ProgramGraphNode *Succ : Node->Successors) {
    if (Succ->Index == 0) {
      strongConnect(Succ);
      Node->LowLink = std::min(Node->LowLink, Succ->LowLink);
    } else if (Succ->OnStack) {
      Node->LowLink = std::min(Node->LowLink, Succ->Index);
    }
  }

  if (Node->LowLink == Node->Index) {
    assert(!SCCStack.empty());
    Postorder.push_back(Node);
    for (;;) {
      auto *N = SCCStack.pop_back_val();
      N->OnStack = false;
      N->Root = Node;
      if (N == Node)
        break;
    }
  }
}

void AutoSeccompModule::coalesceSCCs() {
  for (auto &[_, Node] : ProgramGraphNodes) {
    auto *Root = Node->Root;
    if (Node == Root)
      continue;
    Root->Gen.join(Node->Gen);
    for (auto *Pred : Node->Predecessors) {
      auto *R = Pred->Root;
      if (R && R != Root)
        Root->Predecessors.push_back(R);
    }
    for (auto *Succ : Node->Successors) {
      auto *R = Succ->Root;
      assert(R);
      if (R != Root)
        Root->Successors.push_back(R);
    }
  }

  auto RemoveDuplicates = [](auto &Nodes) {
    std::sort(Nodes.begin(), Nodes.end());
    auto *Last = std::unique(Nodes.begin(), Nodes.end());
    Nodes.erase(Last, Nodes.end());
  };
  for (auto &[_, Node] : ProgramGraphNodes) {
    if (Node == Node->Root) {
      RemoveDuplicates(Node->Predecessors);
      RemoveDuplicates(Node->Successors);
    }
  }
}

void AutoSeccompModule::propagateSyscallSet() {
  for (ProgramGraphNode *Node : Postorder) {
    SyscallNumberSet Set = Node->Gen;
    for (ProgramGraphNode *Succ : Node->Successors) {
      Set.join(Succ->Gen);
    }
    Node->Gen = Set;
  }
}

void AutoSeccompModule::replaceRestrictIntrinsics() {
  LLVMContext &C = M.getContext();
  FunctionType *FT =
      FunctionType::get(Type::getVoidTy(C), Type::getInt8PtrTy(C), false);
  Function *AutoSeccompRestrict = Function::Create(
      FT, Function::ExternalLinkage, "__autoseccomp_restrict", M);

  for (Function &F : M) {
    for (BasicBlock &BB : F) {
      auto I = ++BB.rbegin();
      if (I == BB.rend())
        continue;

      auto *Int = dyn_cast<IntrinsicInst>(&*I);
      if (!Int || Int->getIntrinsicID() != Intrinsic::autoseccomp_restrict)
        continue;

      auto *Node = getProgramGraphNode(BB);
      auto *Root = Node->Root;
      assert(Root);
      DEBUG(errs() << Root->Gen << '\n');

      Value *Syscalls = createSyscallsBitVector(Root, Node->ID);
      ArrayType *ArrayTy =
          ArrayType::get(Type::getInt8Ty(C), SyscallNumberSet::MaxNumber / 8);
      auto *Zero = llvm::ConstantInt::get(llvm::Type::getInt64Ty(C), 0);
      auto *GEP =
          GetElementPtrInst::Create(ArrayTy, Syscalls, {Zero, Zero}, "", Int);
      auto *Call = CallInst::Create(AutoSeccompRestrict, GEP);
      ReplaceInstWithInst(Int, Call);
    }
  }
}

Value *AutoSeccompModule::createSyscallsBitVector(const ProgramGraphNode *Node,
                                                  unsigned ID) {
  LLVMContext &C = M.getContext();

  const SyscallNumberSet &Set = Node->Gen;

  std::array<Constant *, SyscallNumberSet::MaxNumber / 8> Elems;
  Type *Int8Ty = Type::getInt8Ty(C);
  for (unsigned Nr = 0; Nr < SyscallNumberSet::MaxNumber; Nr += 8) {
    uint8_t Val = 0;
    for (unsigned I = 0; I < 8; ++I) {
      if (Set.contains(Nr + I))
        Val |= 1 << I;
    }
    Elems[Nr / 8] = Constant::getIntegerValue(Int8Ty, APInt(8, Val));
  }

  std::string Name;
  raw_string_ostream OS(Name);
  OS << "__autoseccomp_" << Node->BB.getParent()->getName() << "_" << ID;
  OS.flush();

  ArrayType *ArrayTy = ArrayType::get(Int8Ty, SyscallNumberSet::MaxNumber / 8);
  GlobalVariable *G = cast<GlobalVariable>(M.getOrInsertGlobal(Name, ArrayTy));
  Constant *InitVal = ConstantArray::get(ArrayTy, Elems);
  G->setInitializer(InitVal);
  return G;
}

void AutoSeccompModule::deleteTypeIntrinsics() {
  for (Function &F : M) {
    for (BasicBlock &BB : F) {
      auto I = ++BB.rbegin();
      if (I == BB.rend())
        continue;

      auto *Int = dyn_cast<IntrinsicInst>(&*I);
      if (!Int || Int->getIntrinsicID() != Intrinsic::autoseccomp_type)
        continue;

      auto *CastedCallee = dyn_cast<CastInst>(Int->getOperand(0));

      assert(Int->use_empty());
      Int->eraseFromParent();

      if (CastedCallee && CastedCallee->use_empty())
        CastedCallee->eraseFromParent();
    }
  }
}

void AutoSeccompModule::dumpCallGraph(StringRef Filename) const {
  std::error_code EC;
  raw_fd_ostream OS(Filename, EC);
  if (EC) {
    errs() << "Opening '" << Filename << "' failed\n";
    return;
  }

  OS << "digraph CallGraph {\n";
  OS << "nslimit=5\n";

  DenseMap<const Function *, unsigned> FuncIDs;

  unsigned ID = 0;
  for (const Function &F : M) {
    ++ID;
    FuncIDs[&F] = ID;
    OS << "n" << ID << "[label=\"" << F.getName() << "\"]\n";
  }

  ID = 0;
  for (const Function &F : M) {
    ++ID;
    const CallGraphNode *Node = getCallGraphNode(F);
    for (const CallGraphNode *Succ : Node->Successors) {
      unsigned CalleeId = FuncIDs[&Succ->F];
      OS << "n" << ID << "->n" << CalleeId << '\n';
    }
  }

  OS << "}\n";
  OS.flush();
}

void AutoSeccompModule::dumpProgramGraph(bool Coalesced,
                                         StringRef Filename) const {
  std::error_code EC;
  raw_fd_ostream OS(Filename, EC);
  if (EC) {
    errs() << "Opening '" << Filename << "' failed\n";
    return;
  }

  OS << "digraph ProgramGraph {\n";
  OS << "nslimit=5\n";
  for (const auto &[_, N] : ProgramGraphNodes) {
    if (Coalesced && N != N->Root)
      continue;
    const StringRef FuncName = N->BB.getParent()->getName();
    OS << "n" << N->ID << "[label=\"" << FuncName << ":" << N->ID;
    if (!N->Gen.empty())
      OS << "\\n" << N->Gen;
    OS << "\"]\n";
  }
  for (const auto &[_, N] : ProgramGraphNodes) {
    if (Coalesced && N != N->Root)
      continue;
    for (const ProgramGraphNode *Succ : N->Successors) {
      OS << "n" << N->ID << "->n" << Succ->ID;
      if (ProgramGraphNode::isEdgeInterprocedural(N, Succ))
        OS << "[style=dotted]";
      OS << '\n';
    }
  }
  OS << "}\n";
  OS.flush();
}

} // namespace
