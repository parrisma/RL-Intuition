Function BuildDockerFile
{
    [cmdletbinding()]
    Param (
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Target,
        [Parameter(Mandatory = $true)][String]$Location,
        [Boolean]$Interactive = $false,
        [Boolean]$NoCache = $false,
        [Boolean]$PushToHub = $false,
        [Boolean]$Push = $false,
        [Boolean]$Passive = $false,
        [String]$Indent = ""
    )
    Process {
        New-Variable -Name _Res -Value "" -Scope Local
        New-Variable -Name _No_cache -Value "" -Scope Local
        New-Variable -Name _Interactive -Value "" -Scope Local

        if ($NoCache)
        {
            $_No_cache = "--no-cache"
        }
        $_Interactive = ""
        if ($Interactive)
        {
            $_Interactive = "--build-arg interactive_build"
        }

        DockerBuild -Source $Source -Target $Target -Location $Location -NoCache $_No_cache -Interactive $_Interactive -Indent $Indent -Passive $Passive
        DockerPush -Target $Target -PushToHub $PushToHub -Push $Push -Passive $Passive -Indent $Indent
        return
    }
}

Function BuildDockerFiles
{
    [cmdletbinding()]
    Param (
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][Array]$BuildFiles,
        [Parameter(Mandatory = $true)][String]$Location,
        [Boolean]$Interactive = $false,
        [Boolean]$NoCache = $false,
        [Boolean]$PushToHub = $false,
        [Boolean]$Passive = $false,
        [String]$Indent = "   "
    )
    Process {
        if ($BuildFiles.Count -eq 0)
        {
            Write-Output "`r`n$Indent - - - - < Nothing to do"
        }
        else
        {
            New-Variable -Name _Source -Value $null -Scope Local
            New-Variable -Name _Rep -Value $null -Scope Local
            New-Variable -Name _Target -Value $null -Scope Local
            New-Variable -Name _Ver -Value $null -Scope Local
            New-Variable -Name _Push -Value $null -Scope Local
            Foreach ($build_file in $BuildFiles)
            {
                Write-Output "`r`n$Indent - - - - < Processing: [$build_file] - - - - -"
                $build_options = Import-Csv -path $build_file
                Foreach ($line in $build_options)
                {
                    $_Source = $line | Select-Object -ExpandProperty "Dockerfile"
                    $_Rep = $line | Select-Object -ExpandProperty "Repository"
                    $_Target = $line | Select-Object -ExpandProperty "Target"
                    $_Ver = $line | Select-Object -ExpandProperty "Version"
                    $_Push = $line | Select-Object -ExpandProperty "Push"
                    $_Push = StrToBool -Arg $_Push
                    BuildDockerFile -Source $_Source -Location $Location -Target "${_Rep}/${_Target}:${_Ver}" -Interactive $Interactive -NoCache $NoCache -Indent "$Indent   " -Passive $Passive -Push $_Push -PushToHub $PushToHub
                }
            }
        }
        Write-Output "$Indent - - - - > Done: [$build_file] - - - - -`r`n"
        return
    }
}

Function GetBuildFileNames
{
    [cmdletbinding()]
    Param (
        [Parameter(Mandatory = $true)][String]$BuildDir,
        [String]$Filter = "build-*.csv"
    )
    Process {
        New-Variable -Name _Res -Value $null -Scope Local
        New-Variable -Name _File -Value $null -Scope Local
        $_Res = @()
        $_Files = Get-ChildItem -Path $BuildDir -Filter $Filter
        foreach ($f in $_Files)
        {
            $_Res += $f.FullName
        }
        return ,$_Res
    }
}

Function GetAllSubDirectories
{
    [cmdletbinding()]
    Param (
        [Parameter(Mandatory = $true)][String]$RootDir
    )
    Process {
        New-Variable -Name _Res -Value $null -Scope Local
        New-Variable -Name _Dirs -Value $null -Scope Local
        $_Res = @()
        $_Dirs = Get-ChildItem -Directory -Recurse $RootDir
        foreach ($d in $_Dirs)
        {
            $_Res += $d.FullName
        }
        return ,$_Res
    }
}

Function DockerBuild
{
    [cmdletbinding()]
    Param (
        [Parameter(Mandatory = $true)][String]$Source,
        [Parameter(Mandatory = $true)][String]$Target,
        [Parameter(Mandatory = $true)][String]$Location,
        [Parameter(Mandatory = $true)][AllowEmptyString()][String]$NoCache,
        [Parameter(Mandatory = $true)][AllowEmptyString()][String]$Interactive,
        [Boolean]$Passive = $false,
        [String]$Indent = ""
    )
    Process {
        New-Variable -Name _Cmd -Value "" -Scope Local
        New-Variable -Name _Err -Value "" -Scope Local

        Write-Output "$Indent . . . . . Building : [$Source] . . . . ."
        $_Cmd = "docker build -f $Location\$Source $Location -t $Target $Interactive $NoCache"
        Write-Output "$Indent . . . . . [$_Cmd]"

        if (-not$Passive)
        {
            Invoke-Expression $_Cmd *>&1 | Tee-Object -Variable '_Res'
            if ($_Res -match "(.*)Successfully built(.*)(\d+)(.*)")
            {
                Write-Output "$Indent . . . . . Built & Tagged OK: [$Source] to [$Target] . . . . ."
            }
            else
            {
                $_Err = "Docker Build Failed for: [$Source] to [$Target]"
                throw $_Err
            }
        }
        else
        {
            Write-Output "$Indent . . . . . Passive OK . . . . ."
        }
    }
}

Function DockerPush
{
    [cmdletbinding()]
    Param (
        [Parameter(Mandatory = $true)][String]$Target,
        [Parameter(Mandatory = $true)][Boolean]$PushToHub,
        [Parameter(Mandatory = $true)][Boolean]$Push,
        [Boolean]$Passive = $false,
        [String]$Indent = ""
    )
    Process {
        if ($PushToHub -AND $Push)
        {
            New-Variable -Name _Cmd -Value "" -Scope Local
            New-Variable -Name _Err -Value "" -Scope Local

            Write-Output "$Indent . . . . . Pushing : [$Source] . . . . ."
            $_Cmd = "docker push $Target"
            Write-Output "$Indent . . . . . [$_Cmd]"

            if (-Not$Passive)
            {
                Invoke-Expression $_Cmd *>&1 | Tee-Object -Variable '_Res'
                if ($_Res -match "(.*)digest: sha256:(.*)(\d+)(.*)")
                {
                    Write-Output "$Indent . . . . . Built & Tagged OK: [$Source] to [$Target] . . . . ."
                }
                else
                {
                    $_Err = "Docker Build Failed for: [$Source] to [$Target]"
                    throw $_Err
                }
            }
            else
            {
                Write-Output "$Indent . . . . . Passive OK . . . . ."
            }
        }
    }
}

Function DockerNetworkCreate
{
    [cmdletbinding()]
    Param (
        [Parameter(Mandatory = $true)][String]$NetworkName,
        [String]$NetworkType = "bridge",
        [Boolean]$Passive = $false
    )
    Process {
        New-Variable -Name _Cmd -Value "" -Scope Local

        Write-Output "Checking for network [$NetworkName]"
        $_Cmd = "docker network ls -f name=$NetworkName -q"
        Write-Output "Running: [$_Cmd]"

        if (-Not$Passive)
        {
            $_Res = Invoke-Expression $_Cmd
            if ($null -eq $_Res)
            {
                Write-Output "Network does not exist, creating"
                $_Cmd = "docker network create -d $NetworkType --attachable $NetworkName"
                Write-Output "Running: [$_Cmd]"
                $_Res = Invoke-Expression $_Cmd
            }
            else
            {
                Write-Output "Network exists, OK"
            }
        }
    }
}

Function DockerStopContainerByName
{
    [cmdletbinding()]
    Param (
        [Parameter(Mandatory = $true)][String]$ContainerName,
        [Boolean]$Passive = $false
    )
    Process {
        New-Variable -Name _Cmd -Value "" -Scope Local
        New-Variable -Name _Res -Value "" -Scope Local

        Write-Output "Checking for container with name [$ContainerName]"
        $_Cmd = "docker ps --filter name=$ContainerName -q"
        Write-Output "Running: [$_Cmd]"

        if (-Not$Passive)
        {
            $_Res = Invoke-Expression $_Cmd
            if ($null -eq $_Res)
            {
                Write-Output "No instance of a Container with name [$ContainerName]"
            }
            else
            {
                $_Cmd = "docker stop $_Res"
                Write-Output "Running: [$_Cmd]"
                $_Res = Invoke-Expression $_Cmd
            }
        }
    }
}

Function DockerPruneContainersAndImages
{
    [cmdletbinding()]
    Param (
        [Boolean]$Passive = $false
    )
    Process {
        if (-Not$Passive)
        {
            docker image prune
            docker container prune --force
        }
    }
}

Function DockerIsSwarmManager
{
    [cmdletbinding()]
    Param (
        [Boolean]$Passive = $false
    )
    Process {
        New-Variable -Name _Cmd -Value "" -Scope Local
        New-Variable -Name _Res -Value "" -Scope Local

        Write-Output "Checking for swarm manager status for localhost"
        $_Cmd = "docker info --format '{{json .Swarm.ControlAvailable}}'"
        Write-Output "Running: [$_Cmd]"

        if (-Not$Passive)
        {
            $_Res = Invoke-Expression $_Cmd
            $_Res = StrToBool -Arg $_Res
        }
        return $_Res
    }
}

Function DockerStartStack
{
    [cmdletbinding()]
    Param (
        [Parameter(Mandatory = $true)][String]$ComposeYML,
        [Parameter(Mandatory = $true)][String]$StackName,
        [Boolean]$Passive = $false
    )
    Process {
        New-Variable -Name _Cmd -Value "" -Scope Local
        New-Variable -Name _Res -Value "" -Scope Local

        Write-Output "Starting docker stack [$StackName] from [$ComposeYML]"
        $_Cmd = "docker stack deploy -c $ComposeYML $StackName"
        Write-Output "Running: [$_Cmd]"

        if (-Not$Passive)
        {
            $_Res = Invoke-Expression $_Cmd
            Write-Output $_Res
        }
        return
    }
}

Function StrToBool
{
    [cmdletbinding()]
    Param (
        [Parameter(Mandatory = $true)][String]$Arg
    )
    Process {
        New-Variable -Name _Res -Value $false -Scope Local
        $Arg = $Arg.ToLower()
        if ($Arg -eq "true" -OR $Arg -eq "1" -OR $Arg -eq "`$true" -OR $Arg -eq "yes")
        {
            $_Res = $true
        }
        return $_Res
    }
}