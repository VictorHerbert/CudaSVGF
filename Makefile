# ===========================================================================
#                           Compiler and Flags
# ===========================================================================

NVCC = nvcc
CXX = $(NVCC)

CXXFLAGS_LK = -w -lineinfo -O3 --use_fast_math -rdc=true -std=c++17 -arch=sm_75 -I./$(INCLUDE_DIR) 
#CXXFLAGS_LK = -w -G -g -rdc=true -std=c++17 -arch=sm_75 -I./$(INCLUDE_DIR)
CXXFLAGS = $(CXXFLAGS_LK) -dc

CSAN_TOOLS := memcheck racecheck synccheck initcheck

MKDIR = mkdir
RM = rm

# ===========================================================================
#                               Directories
# ===========================================================================

SRC_DIR     = src
BUILD_DIR   = build
TEST_DIR   	= test
INCLUDE_DIR = src/include

# ===========================================================================
#                        Source and Object Files
# ===========================================================================

SRC			= $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_DIR)/*.cu)
OBJ			= $(SRC:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
OBJ			:= $(OBJ:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)
DEPS		= $(OBJ:.o=.d)

-include $(DEPS)

# ===========================================================================
#                                 Target
# ===========================================================================

TARGET = $(BUILD_DIR)/cudasvgf

# ===========================================================================
#                               Build Rules
# ===========================================================================

$(TARGET): $(OBJ)
	@echo "Linking $^ into $@"
	$(NVCC) $(CXXFLAGS_LK) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@printf "Recompiling %-20s into %s\n" "$<" "$@"
	@$(NVCC) $(CXXFLAGS) -M -MT $@ $< > $(BUILD_DIR)/$*.d
	$(NVCC) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@printf "Recompiling %-20s into %s\n" "$<" "$@"
	@$(NVCC) $(CXXFLAGS) -M -MT $@ $< > $(BUILD_DIR)/$*.d
	$(NVCC) $(CXXFLAGS) -c $< -o $@


# ===========================================================================
#                                 Tasks
# ===========================================================================

compile: $(TARGET)

test: $(TARGET)
	@./$(TARGET) -t

cli: $(TARGET)
	@./$(TARGET) render/cornell/ 512 512 3 10 .01 .01

check: $(CSAN_TOOLS)

$(CSAN_TOOLS): $(TARGET) 
	@compute-sanitizer --tool $@ --show-backtrace=yes --log-file $(TEST_DIR)/$@.log ./$(TARGET) -t

doxygen:
	doxygen Doxyfile

all: memcheck doxygen

.PHONY: test $(CSAN_TOOLS) check ncu doxygen 

# ===========================================================================
#                                  Clean
# ===========================================================================

test_clean: $(TARGET)
	@$(RM) -rf test/*
	@$(RM) -rf render/sponza/output/*	

clean:
	$(RM) -rf $(BUILD_DIR)/*
	$(RM) -rf test/*
	$(MKDIR) -p test