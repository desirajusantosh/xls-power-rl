// Fast RTL toggle-count proxy for an XLS streaming proc, driven over stdin/stdout.
//
// Build: see xpr/proxy.py. The model is compiled with --public-flat-rw so every
// RTL variable is reachable through Verilator's scope tables; toggles are counted
// by XOR-ing a byte snapshot of all variables (except clk) after each cycle.
//
// Protocol, one request per line:
//   <n_cycles> <win_start> <w_0> ... <w_{n-1}>     words are hex {rst, vld, data}
// One response line per request:
//   <toggles> <bits> <window_cycles> <vld_not_rdy_cycles> <out_vld_cycles>
// With PROXY_PERCYCLE set, the line continues with " |" and one toggle count per
// window cycle (the input change at the start of the cycle plus its rising edge).
// A fresh model is built for every request, so no state carries over.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <algorithm>
#include <cstdlib>
#include <set>

#include "Vdut.h"  // built with --prefix Vdut
#include "verilated.h"
#include "verilated_syms.h"

struct Region {
  uint8_t* data;
  size_t bytes;  // storage, including padding (padding bits never toggle)
  size_t bits;   // declared width x unpacked elements: the alpha denominator
};

static size_t declared_bits(const VerilatedVar& v) {
  size_t bits = v.packed().elements();
  // left(d)/right(d): d = 0 is the packed range, 1..udims() the unpacked ones.
  for (int d = 1; d <= v.udims(); ++d) bits *= std::abs(v.left(d) - v.right(d)) + 1;
  return bits;
}

// Every variable of the DUT's top module except clocks (same rule as
// xpr.gatesim.is_clock_net). Only the wrapper scope ("TOP") and the DUT scope
// ("TOP.<dut>") count: in a gate-level netlist, deeper scopes are cell
// internals, which the Icarus VCD ($dumpvars(1, dut)) does not include either.
// Ports appear in both counted scopes with separate storage, so de-duplicate
// by name.
static std::vector<Region> collect(VerilatedContext* ctx) {
  std::vector<Region> regions;
  std::set<std::string> seen;
  const bool debug = std::getenv("PROXY_DEBUG") != nullptr;
  const VerilatedScopeNameMap* scopes = ctx->scopeNameMap();
  if (!scopes) return regions;
  for (const auto& sit : *scopes) {
    const std::string scope = sit.first;
    if (std::count(scope.begin(), scope.end(), '.') > 1) continue;
    const VerilatedVarNameMap* vars = sit.second->varsp();
    if (!vars) continue;
    for (const auto& vit : *vars) {
      std::string name = vit.first;
      std::string lower = name;
      for (auto& ch : lower) ch = std::tolower(ch);
      const VerilatedVar& v = vit.second;
      if (lower.find("clk") != std::string::npos || !seen.insert(name).second) continue;
      if (debug) std::cerr << sit.first << " . " << name << " bits=" << declared_bits(v) << "\n";
      regions.push_back({static_cast<uint8_t*>(v.datap()), static_cast<size_t>(v.totalSize()), declared_bits(v)});
    }
  }
  return regions;
}

static size_t total_bits(const std::vector<Region>& r) {
  size_t b = 0;
  for (const auto& x : r) b += x.bits;
  return b;
}

int main(int argc, char** argv) {
  std::ios::sync_with_stdio(false);
  std::string line;
  bool printed_bits = false;
  const bool per_cycle = std::getenv("PROXY_PERCYCLE") != nullptr;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    std::istringstream in(line);
    long n, win_start;
    in >> n >> win_start;
    std::vector<uint64_t> words(n);
    for (long i = 0; i < n; ++i) in >> std::hex >> words[i];

    auto ctx = std::make_unique<VerilatedContext>();
    ctx->commandArgs(argc, argv);
    auto top = std::make_unique<Vdut>(ctx.get());
    std::vector<Region> regions = collect(ctx.get());
    if (!printed_bits) {
      std::cerr << "proxy: " << regions.size() << " variables, " << total_bits(regions) << " bits\n";
      printed_bits = true;
    }
    size_t bytes = 0;
    for (const auto& r : regions) bytes += r.bytes;
    std::vector<uint8_t> prev(bytes), cur(bytes);
    auto snap = [&](std::vector<uint8_t>& dst) {
      size_t o = 0;
      for (const auto& r : regions) { std::memcpy(dst.data() + o, r.data, r.bytes); o += r.bytes; }
    };

    uint64_t toggles = 0, vld_not_rdy = 0, out_vld = 0;
    std::vector<uint64_t> cyc(n > win_start ? n - win_start : 0, 0);
    long cur_cycle = 0;
    const uint64_t data_mask = (IN_W >= 64) ? ~0ull : ((1ull << IN_W) - 1);
    // Matches the Icarus testbench: inputs change with the clock low, flops update
    // on the rising edge. A net can change at both points in one cycle, so compare
    // snapshots after each settle. The baseline is the settle after win_start's
    // inputs, where the testbench starts its VCD.
    auto count = [&]() {
      snap(cur);
      uint64_t t = 0;
      for (size_t i = 0; i < bytes; ++i) t += __builtin_popcount(prev[i] ^ cur[i]);
      toggles += t;
      cyc[cur_cycle] += t;
      prev.swap(cur);
    };
    top->clk = 0;
    top->_out_s_rdy = 1;
    for (long c = 0; c < n; ++c) {
      uint64_t w = words[c];
      top->rst = (w >> (IN_W + 1)) & 1;
      top->_in_r_vld = (w >> IN_W) & 1;
      top->_in_r = w & data_mask;
      top->clk = 0;
      top->eval();
      cur_cycle = c - win_start;
      if (c == win_start) snap(prev);
      if (c > win_start) count();
      if (c >= win_start) {
        if (top->_in_r_vld && !top->_in_r_rdy) ++vld_not_rdy;
        if (top->_out_s_vld) ++out_vld;
      }
      top->clk = 1;
      top->eval();
      if (c >= win_start) count();
    }
    top->final();
    std::cout << toggles << ' ' << total_bits(regions) << ' ' << (n - win_start) << ' ' << vld_not_rdy << ' '
              << out_vld;
    if (per_cycle) {
      std::cout << " |";
      for (uint64_t t : cyc) std::cout << ' ' << t;
    }
    std::cout << '\n';
    std::cout.flush();
  }
  return 0;
}
