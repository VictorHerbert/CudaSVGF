# ===========================================================================
#                           Compiler and Flags
# ===========================================================================

NVCC = nvcc
CXX = $(NVCC)

CXXFLAGS_LK = -w -G -g -O3 -rdc=true -std=c++17 -arch=sm_75 -I./include
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
INCLUDE_DIR = include

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

TARGET = $(BUILD_DIR)/main

# ===========================================================================
#                               Build Rules
# ===========================================================================

$(TARGET): $(OBJ)
	@echo "Linking $^ into $@"
	@$(NVCC) $(CXXFLAGS_LK) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@printf "Recompiling %-15s into %s\n" "$<" "$@"
	@$(NVCC) $(CXXFLAGS) -M -MT $@ $< > $(BUILD_DIR)/$*.d
	@$(NVCC) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@printf "Recompiling %-15s into %s\n" "$<" "$@"
	@$(NVCC) $(CXXFLAGS) -M -MT $@ $< > $(BUILD_DIR)/$*.d
	@$(NVCC) $(CXXFLAGS) -c $< -o $@


# ===========================================================================
#                                 Tasks
# ===========================================================================

test: $(TARGET)
	@./$(TARGET) -t

run_no_args: $(TARGET)
	@./$(TARGET)

$(CSAN_TOOLS): $(TARGET) 
	@compute-sanitizer --tool $@ --show-backtrace=yes --log-file $(TEST_DIR)/$@.log ./$(TARGET) -t

ncu: $(TARGET)
	@ncu ./$(TARGET) -t

doxygen:
	doxygen Doxyfile

all: memcheck doxygen

.PHONY: test $(CSAN_TOOLS) ncu doxygen

# ===========================================================================
#                                  Clean
# ===========================================================================

test_clean: $(TARGET)
	@$(RM) -rf test/*
	@./$(TARGET) -t

clean:
	$(RM) -rf $(BUILD_DIR)/*
	$(RM) -rf test/*
	$(MKDIR) -p test