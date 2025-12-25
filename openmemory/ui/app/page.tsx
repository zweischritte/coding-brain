"use client";

import Stats from "@/components/dashboard/Stats";
import { GraphStats } from "@/components/dashboard/GraphStats";
import { MemoryFilters } from "@/app/memories/components/MemoryFilters";
import { MemoriesSection } from "@/app/memories/components/MemoriesSection";
import "@/styles/animation.css";

export default function DashboardPage() {
  return (
    <div className="text-white py-6">
      <div className="container">
        <div className="w-full mx-auto space-y-6">
          <div className="grid grid-cols-3 gap-6">
            {/* Memory Stats */}
            <div className="col-span-2 animate-fade-slide-down">
              <Stats />
            </div>

            {/* Graph Stats */}
            <div className="col-span-1 animate-fade-slide-down delay-1">
              <GraphStats />
            </div>
          </div>

          <div>
            <div className="animate-fade-slide-down delay-2">
              <MemoryFilters />
            </div>
            <div className="animate-fade-slide-down delay-3">
              <MemoriesSection />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
