/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "ucc_global_opts.h"
#include "utils/ucc_compiler_def.h"

ucc_global_config_t ucc_global_config = {
    .log_component  = {UCC_LOG_LEVEL_WARN, "UCC"},
    .component_path = "",
    .initialized    = 0,
};

ucc_config_field_t ucc_global_config_table[] = {
    {"LOG_LEVEL", "warn",
     "UCC logging level. Messages with a level higher or equal to the selected "
     "will be printed.\n"
     "Possible values are: fatal, error, warn, info, debug, trace, data, func, "
     "poll.",
     ucc_offsetof(ucc_global_config_t, log_component),
     UCC_CONFIG_TYPE_LOG_COMP},

    {"COMPONENT_PATH", "", "Specifies dynamic components location",
     ucc_offsetof(ucc_global_config_t, component_path), UCC_CONFIG_TYPE_STRING},

    {NULL}
};

UCC_CONFIG_REGISTER_TABLE(ucc_global_config_table, "UCC global", NULL,
                          ucc_global_config)
