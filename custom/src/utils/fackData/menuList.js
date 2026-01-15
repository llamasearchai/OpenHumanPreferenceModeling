export const menuList = [
    {
        id: 0,
        name: "overview",
        path: "#",
        icon: 'feather-activity',
        dropdownMenu: [
            {
                id: 1,
                name: "Unified Dashboard",
                path: "/",
                subdropdownMenu: false
            }
        ]
    },
    {
        id: 1,
        name: "modules",
        path: "#",
        icon: 'feather-layers',
        dropdownMenu: [
            {
                id: 1,
                name: "Modules Home",
                path: "/modules",
                subdropdownMenu: false
            },
            {
                id: 2,
                name: "Components Gallery",
                path: "/modules/components",
                subdropdownMenu: false
            },
            {
                id: 3,
                name: "Biomarker Analysis Agent",
                path: "/modules/biomarker-analysis",
                subdropdownMenu: false
            },
            {
                id: 4,
                name: "Biomarker Matrices",
                path: "/modules/biomarker-matrices",
                subdropdownMenu: false
            },
            {
                id: 5,
                name: "Bioprocess Automation",
                path: "/modules/bioprocess-automation",
                subdropdownMenu: false
            },
            {
                id: 6,
                name: "Genomic Sequencing",
                path: "/modules/genomic-sequencing",
                subdropdownMenu: false
            },
            {
                id: 7,
                name: "Imaging Agent",
                path: "/modules/imaging-agent",
                subdropdownMenu: false
            }
        ]
    },
    {
        id: 2,
        name: "operations",
        path: '#',
        icon: 'feather-clipboard',
        dropdownMenu: [
            {
                id: 1,
                name: "Chat Ops",
                path: "/applications/chat",
                subdropdownMenu: false
            },
            {
                id: 2,
                name: "Email Console",
                path: "/applications/email",
                subdropdownMenu: false
            },
            {
                id: 3,
                name: "Tasks",
                path: "/applications/tasks",
                subdropdownMenu: false
            },
            {
                id: 4,
                name: "Notes",
                path: "/applications/notes",
                subdropdownMenu: false
            },
            {
                id: 5,
                name: "Storage",
                path: "/applications/storage",
                subdropdownMenu: false
            },
            {
                id: 6,
                name: "Calendar",
                path: "/applications/calendar",
                subdropdownMenu: false
            },
        ]
    },
    {
        id: 3,
        name: "authentication",
        path: "#",
        icon: 'feather-power',
        dropdownMenu: [
            {
                id: 1,
                name: "login",
                path: "#",
                subdropdownMenu: [
                    {
                        id: 1,
                        name: "Cover",
                        path: "/authentication/login/cover",
                    },
                    {
                        id: 2,
                        name: "Minimal",
                        path: "/authentication/login/minimal",
                    },
                    {
                        id: 3,
                        name: "Creative",
                        path: "/authentication/login/creative",
                    },
                ]
            },
            {
                id: 2,
                name: "register",
                path: "#",
                subdropdownMenu: [
                    {
                        id: 1,
                        name: "Cover",
                        path: "/authentication/register/cover",
                    },
                    {
                        id: 2,
                        name: "Minimal",
                        path: "/authentication/register/minimal",
                    },
                    {
                        id: 3,
                        name: "Creative",
                        path: "/authentication/register/creative",
                    },
                ]
            },
            {
                id: 3,
                name: "Error 404",
                path: "#",
                subdropdownMenu: [
                    {
                        id: 1,
                        name: "Cover",
                        path: "/authentication/404/cover",
                    },
                    {
                        id: 2,
                        name: "Minimal",
                        path: "/authentication/404/minimal",
                    },
                    {
                        id: 3,
                        name: "Creative",
                        path: "/authentication/404/creative",
                    },
                ]
            },
            {
                id: 4,
                name: "Reset Pass",
                path: "#",
                subdropdownMenu: [
                    {
                        id: 1,
                        name: "Cover",
                        path: "/authentication/reset/cover",
                    },
                    {
                        id: 2,
                        name: "Minimal",
                        path: "/authentication/reset/minimal",
                    },
                    {
                        id: 3,
                        name: "Creative",
                        path: "/authentication/reset/creative",
                    },
                ]
            },
            {
                id: 5,
                name: "Verify OTP",
                path: "#",
                subdropdownMenu: [
                    {
                        id: 1,
                        name: "Cover",
                        path: "/authentication/verify/cover",
                    },
                    {
                        id: 2,
                        name: "Minimal",
                        path: "/authentication/verify/minimal",
                    },
                    {
                        id: 3,
                        name: "Creative",
                        path: "/authentication/verify/creative",
                    },
                ]
            },
            {
                id: 6,
                name: "Maintenance",
                path: "#",
                subdropdownMenu: [
                    {
                        id: 1,
                        name: "Cover",
                        path: "/authentication/maintenance/cover",
                    },
                    {
                        id: 2,
                        name: "Minimal",
                        path: "/authentication/maintenance/minimal",
                    },
                    {
                        id: 3,
                        name: "Creative",
                        path: "/authentication/maintenance/creative",
                    },
                ]
            },
        ]
    },
]
